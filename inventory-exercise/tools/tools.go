package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
)

type Tool struct {
	Name        string
	Description string
	Parameters  openai.FunctionParameters // 底层就是 map[string]any
	Exec        func(args json.RawMessage) (string, error)
}

// Report 是盘点报告的强类型 schema，write_report 落盘前用它做校验。
// json tag 的 key 必须和任务描述里声明的字段名逐字一致（见下面的坑）。
type Report struct {
	Warehouse      string   `json:"warehouse"`       // 仓库
	TotalItems     int      `json:"total_items"`     // 商品种数
	TotalQuantity  int      `json:"total_quantity"`  // 总件数
	UncheckedItems []string `json:"unchecked_items"` // 未盘点商品列表 ← 考核2的 array
	IsComplete     bool     `json:"is_complete"`     // 是否全部完成 ← 考核2的 boolean
}

// 题目要求：库存数据故意指向不存在的 stock.txt（错误线索），
// 模型读到 file-not-found 后必须通过 list_files 自救。
var fileMap = map[string]string{
	"库存": "./data/stock.txt",
	"团队": "./data/team.txt",
}

// registry 保存全部工具定义，BuildTools 供模型选择，Dispatch 供 agent loop 执行。
var registry []Tool

// safeDataPath 路径安全：拒绝绝对路径、拒绝 .. 穿越，只允许访问 data/ 目录。
func safeDataPath(p string) (string, error) {
	if filepath.IsAbs(p) {
		return "", errors.New("absolute paths are not allowed")
	}
	clean := filepath.Clean(p) // "./data/x.txt" -> "data/x.txt"
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", errors.New("path traversal is not allowed")
	}
	if clean != "data" && !strings.HasPrefix(clean, "data"+string(filepath.Separator)) {
		return "", errors.New("only the data/ directory is accessible")
	}
	return clean, nil
}

// 构建tools
func buildTools(tools []Tool) []responses.ToolUnionParam {
	ret := make([]responses.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		ret = append(ret, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Strict:      openai.Bool(true),
				Parameters:  t.Parameters,
				Name:        t.Name,
				Description: openai.String(t.Description),
			},
		})
	}
	return ret
}

func search(data string) (string, error) {
	for d, path := range fileMap {
		if strings.Contains(d, data) || strings.Contains(data, d) {
			return path, nil
		}
	}
	return "", errors.New("data file not found,(supported topics: 库存, 团队)")
}

func read_file(path string) (string, error) {
	safe, err := safeDataPath(path)
	if err != nil {
		return "", err
	}
	b, err := os.ReadFile(safe)
	if err != nil {
		if os.IsNotExist(err) {
			// 错误信息里带上自愈提示，引导模型改用 list_files
			return "", fmt.Errorf("file not found: %s (try list_files to see available files)", safe)
		}
		return "", err
	}
	return string(b), nil
}

func list_files() (string, error) {
	entries, err := os.ReadDir("data")
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	for _, e := range entries {
		// 直接返回 data/ 前缀的完整路径，模型拿到后可直接传给 read_file，
		// 避免把裸文件名传进去触发路径防护（只允许 data/ 目录）
		sb.WriteString("data/")
		sb.WriteString(e.Name())
		sb.WriteString("\n")
	}
	return sb.String(), nil
}

func calculator(products map[string]int64) string {
	var total int64
	for _, n := range products {
		total += n
	}
	// 输出 key 与 Report 的 json tag 同名同义（total_items=商品种数, total_quantity=总件数），
	// 模型拿到后可直接照抄进 write_report 的报告，零翻译成本
	return fmt.Sprintf(`{"total_quantity": %d, "total_items": %d}`, total, len(products))
}

func write_report(result string) (string, error) {
	var report Report
	// Unmarshal 一步完成两件事：语法校验 + 类型校验（比 json.Valid 更严）
	if err := json.Unmarshal([]byte(result), &report); err != nil {
		return "", fmt.Errorf(
			"invalid report JSON (%v); required fields: warehouse(仓库), total_items(商品种数), total_quantity(总件数), unchecked_items(未盘点商品列表), is_complete(是否全部盘点完成)",
			err)
	}
	// 字段级校验——Unmarshal 对"缺字段"不报错，只给零值，需要手动兜底
	if report.Warehouse == "" || report.TotalItems == 0 || report.TotalQuantity == 0 {
		return "", errors.New("report missing required fields: warehouse, total_items, total_quantity")
	}
	// 未盘点列表缺省时是 nil，MarshalIndent 会落盘成 null，规范化为 []
	if report.UncheckedItems == nil {
		report.UncheckedItems = []string{}
	}

	if len(report.UncheckedItems) > 0 && report.IsComplete {
		return "", errors.New("存在未盘点商品但 is_complete=true，请修正后重新提交")
	}

	safe, err := safeDataPath("data/report.json")
	if err != nil {
		return "", err
	}
	out, err := json.MarshalIndent(report, "", "  ") // 规范化输出，落盘格式统一
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(safe, out, 0o644); err != nil {
		return "", err
	}
	return "report saved to data/report.json", nil
}

// BuildTools 返回提供给模型的工具 schema 列表。
func BuildTools() []responses.ToolUnionParam {
	registry = []Tool{
		{
			Name:        "search",
			Description: "根据主题关键词（如 库存、团队）在内置知识库中查找该主题的数据文件存储路径。读取文件前先调用本工具确认路径",
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"data": map[string]any{
						"type":        "string",
						"description": "主题关键词，例如：库存",
					},
				},
				"required":             []any{"data"},
				"additionalProperties": false,
			},
			Exec: func(args json.RawMessage) (string, error) {
				var a struct {
					Data string `json:"data"`
				}
				if err := json.Unmarshal(args, &a); err != nil {
					return "", err
				}
				return search(a.Data)
			},
		},
		{
			Name:        "read_file",
			Description: "读取文件内容。path 必须是 search 或 list_files 返回的完整相对路径（例如 data/inventory.txt），且只能在 data/ 目录内",
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "要读取的文件路径，例如 data/inventory.txt",
					},
				},
				"required":             []any{"path"},
				"additionalProperties": false,
			},
			Exec: func(args json.RawMessage) (string, error) {
				var a struct {
					Path string `json:"path"`
				}
				if err := json.Unmarshal(args, &a); err != nil {
					return "", err
				}
				return read_file(a.Path)
			},
		},
		{
			Name:        "calculator",
			Description: "统计全部商品的总件数与商品种数。products 必须传入全部商品，未盘点（待盘点）的商品数量传 0，禁止遗漏；返回 {\"total_quantity\": 总件数, \"total_items\": 商品种数}，数值可直接填入报告的同名字段",
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"products": map[string]any{
						"type":        "object",
						"description": "全部商品的名称及数量（未盘点的商品传 0）",
						"additionalProperties": map[string]any{
							"type":        "integer",
							"description": "该商品的库存数量，未盘点（待盘点）传 0",
						},
					},
				},
				"required":             []any{"products"},
				"additionalProperties": false,
			},
			Exec: func(args json.RawMessage) (string, error) {
				var a struct {
					Products map[string]int64 `json:"products"`
				}
				if err := json.Unmarshal(args, &a); err != nil {
					return "", err
				}
				return calculator(a.Products), nil
			},
		},
		{
			Name:        "list_files",
			Description: "列出 data/ 目录下所有数据文件的完整相对路径（每行一个，如 data/inventory.txt）。read_file 读取失败时用本工具查看实际存在的文件名",
			Parameters: openai.FunctionParameters{
				"type":                 "object",
				"properties":           map[string]any{},
				"required":             []any{},
				"additionalProperties": false,
			},
			Exec: func(args json.RawMessage) (string, error) {
				return list_files()
			},
		},
		{
			Name:        "write_report",
			Description: "将盘点报告以 JSON 格式保存到 data/report.json。报告必须包含字段：warehouse(仓库)、total_items(商品种数)、total_quantity(总件数)、unchecked_items(未盘点商品列表)、is_complete(是否全部盘点完成)。任务要求保存报告时必须调用本工具，禁止只在回复文本中给出报告",
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"result": map[string]any{
						"type":        "string",
						"description": "盘点报告的 JSON 字符串",
					},
				},
				"required":             []any{"result"},
				"additionalProperties": false,
			},
			Exec: func(args json.RawMessage) (string, error) {
				var a struct {
					Result string `json:"result"`
				}
				if err := json.Unmarshal(args, &a); err != nil {
					return "", err
				}
				return write_report(a.Result)
			},
		},
	}
	return buildTools(registry)
}

// Dispatch 按工具名称分发执行，供 agent loop 调用。
// 未知工具或执行出错时返回错误字符串，由调用方喂回模型自愈。
func Dispatch(name string, args json.RawMessage) (string, error) {
	for _, t := range registry {
		if t.Name == name {
			return t.Exec(args)
		}
	}
	return "", fmt.Errorf("unknown tool: %s", name)
}
