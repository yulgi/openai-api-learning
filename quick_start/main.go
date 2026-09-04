package main

// 学习成果考核 demo，覆盖六项：
// [1] 用 LLM API 完成普通对话
// [2] 让模型输出结构化 JSON（strict schema + enum）
// [3] 定义工具函数（search / calculator / read_file）
// [4] 解析模型的 tool call / function call
// [5] 执行工具并把结果喂回模型
// [6] agent loop 的最大步数、超时、错误处理

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

const (
	model    = "grok-4.6"
	dataDir  = "data"          // read_file 工具只允许访问该目录（路径穿越防护）
	maxSteps = 6               // [6] agent loop 最大步数，防止无限循环
	budget   = 3 * time.Minute // [6] agent loop 总超时
)

func main() {
	client := openai.NewClient(
		// OpenCode Zen 网关：SDK 会自动拼接 /responses
		option.WithBaseURL("https://opencode.ai/zen/go/v1"),
		// key 从环境变量读取，不要硬编码
		option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)
	ctx := context.Background()

	fmt.Println("========== [1] 普通对话 ==========")
	basicChat(ctx, client)

	fmt.Println("\n========== [2] 结构化 JSON 输出 ==========")
	structuredJSON(ctx, client)

	fmt.Println("\n========== [3] Agent Loop：search -> read_file -> calculator ==========")
	agentLoop(ctx, client)
}

// ---------- [1] 普通对话 ----------

func basicChat(ctx context.Context, client openai.Client) {
	resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: model,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String("你好！请用一句话介绍你自己。"),
		},
	})
	if err != nil {
		fmt.Println("API 错误:", err)
		return
	}
	fmt.Println("模型:", resp.OutputText())
}

// ---------- [2] 结构化 JSON 输出 ----------

type textAnalysis struct {
	Summary   string   `json:"summary"`
	Sentiment string   `json:"sentiment"` // schema 里用 enum 限定取值
	Keywords  []string `json:"keywords"`
}

func structuredJSON(ctx context.Context, client openai.Client) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"summary":   map[string]any{"type": "string", "description": "用一句话概括原文"},
			"sentiment": map[string]any{"type": "string", "enum": []string{"positive", "negative", "neutral"}},
			"keywords": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
		},
		"required":             []string{"summary", "sentiment", "keywords"},
		"additionalProperties": false, // strict 模式硬性要求
	}

	resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: model,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(`分析这段文字："熬了三个通宵终于把 Go 的 agent loop 调通了，头秃了但真的很爽。"`),
		},
		Text: responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   "text_analysis",
					Schema: schema,
					Strict: openai.Bool(true),
				},
			},
		},
	})
	if err != nil {
		fmt.Println("API 错误:", err)
		return
	}
	fmt.Println("模型原始输出:", resp.OutputText())

	var a textAnalysis
	if err := json.Unmarshal([]byte(resp.OutputText()), &a); err != nil {
		fmt.Println("JSON 解析失败:", err)
		return
	}
	fmt.Printf("解析结果 -> 摘要: %s | 情感: %s | 关键词: %v\n", a.Summary, a.Sentiment, a.Keywords)
}

// ---------- [3] 工具实现 ----------

var knowledgeBase = []string{
	"data.txt    —— 本季度项目预算",
	"team.txt    —— 团队成员名单",
	"meeting.txt —— 上周周会纪要",
}

func runSearch(args json.RawMessage) string {
	var a struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(args, &a); err != nil {
		return "参数解析失败: " + err.Error()
	}
	var hits []string
	for _, entry := range knowledgeBase {
		if strings.Contains(entry, a.Query) {
			hits = append(hits, entry)
		}
	}
	if len(hits) == 0 {
		return fmt.Sprintf("没有找到与 %q 相关的记录", a.Query)
	}
	return strings.Join(hits, "\n")
}

func runCalculator(args json.RawMessage) string {
	var a struct {
		Expression string `json:"expression"`
	}
	if err := json.Unmarshal(args, &a); err != nil {
		return "参数解析失败: " + err.Error()
	}
	val, err := evalArithmetic(a.Expression)
	if err != nil {
		return "计算失败: " + err.Error()
	}
	if val == math.Trunc(val) {
		return strconv.FormatInt(int64(val), 10)
	}
	return strconv.FormatFloat(val, 'f', -1, 64)
}

// 用 go/ast 解析并计算算术表达式，不引入 eval，杜绝注入风险
func evalArithmetic(expr string) (float64, error) {
	node, err := parser.ParseExpr(expr)
	if err != nil {
		return 0, fmt.Errorf("表达式语法错误: %w", err)
	}
	return evalNode(node)
}

func evalNode(n ast.Expr) (float64, error) {
	switch v := n.(type) {
	case *ast.ParenExpr:
		return evalNode(v.X)
	case *ast.UnaryExpr:
		val, err := evalNode(v.X)
		if err != nil {
			return 0, err
		}
		if v.Op == token.SUB {
			return -val, nil
		}
		if v.Op == token.ADD {
			return val, nil
		}
		return 0, fmt.Errorf("不支持的一元运算符 %v", v.Op)
	case *ast.BinaryExpr:
		left, err := evalNode(v.X)
		if err != nil {
			return 0, err
		}
		right, err := evalNode(v.Y)
		if err != nil {
			return 0, err
		}
		switch v.Op {
		case token.ADD:
			return left + right, nil
		case token.SUB:
			return left - right, nil
		case token.MUL:
			return left * right, nil
		case token.QUO:
			if right == 0 {
				return 0, errors.New("除数不能为 0")
			}
			return left / right, nil
		default:
			return 0, fmt.Errorf("不支持的运算符 %v", v.Op)
		}
	case *ast.BasicLit:
		return strconv.ParseFloat(v.Value, 64)
	}
	return 0, fmt.Errorf("不支持的表达式节点 %T", n)
}

func runReadFile(args json.RawMessage) string {
	var a struct {
		Filename string `json:"filename"`
	}
	if err := json.Unmarshal(args, &a); err != nil {
		return "参数解析失败: " + err.Error()
	}
	if a.Filename == "" || filepath.IsAbs(a.Filename) ||
		strings.Contains(filepath.ToSlash(filepath.Clean(a.Filename)), "..") {
		return "非法路径：只允许 data 目录下的文件名"
	}
	content, err := os.ReadFile(filepath.Join(dataDir, a.Filename))
	if err != nil {
		return "读取失败: " + err.Error()
	}
	return string(content)
}

// 工具注册表：新增工具只需在这里加一项，agent loop 无需改动
type agentTool struct {
	def responses.ToolUnionParam
	run func(args json.RawMessage) string
}

func buildTools() ([]responses.ToolUnionParam, map[string]agentTool) {
	search := responses.ToolParamOfFunction("search", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{"type": "string", "description": "搜索关键词"},
		},
		"required":             []string{"query"},
		"additionalProperties": false,
	}, true)
	search.OfFunction.Description = openai.String("在公司知识库中搜索资料，返回相关文件名和简介")

	calculator := responses.ToolParamOfFunction("calculator", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"expression": map[string]any{"type": "string", "description": "算术表达式，支持 + - * / 和括号"},
		},
		"required":             []string{"expression"},
		"additionalProperties": false,
	}, true)
	calculator.OfFunction.Description = openai.String("计算算术表达式并返回结果")

	readFile := responses.ToolParamOfFunction("read_file", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"filename": map[string]any{"type": "string", "description": "文件名，如 data.txt（仅限 data 目录内）"},
		},
		"required":             []string{"filename"},
		"additionalProperties": false,
	}, true)
	readFile.OfFunction.Description = openai.String("读取 data 目录下的指定文件内容")

	return []responses.ToolUnionParam{search, calculator, readFile}, map[string]agentTool{
		"search":     {def: search, run: runSearch},
		"calculator": {def: calculator, run: runCalculator},
		"read_file":  {def: readFile, run: runReadFile},
	}
}

// ---------- [4][5][6] 解析 tool call -> 执行 -> 回传，循环带保护 ----------

func agentLoop(ctx context.Context, client openai.Client) {
	ctx, cancel := context.WithTimeout(ctx, budget) // [6] 总超时
	defer cancel()

	toolParams, toolsByName := buildTools()

	task := "公司现在还剩多少项目预算？把金额乘以 15 再加上 100，告诉我最终结果，并说明你参考了哪些资料。"

	input := []responses.ResponseInputItemUnionParam{
		{OfMessage: &responses.EasyInputMessageParam{
			Role:    responses.EasyInputMessageRoleUser,
			Content: responses.EasyInputMessageContentUnionParam{OfString: openai.String(task)},
		}},
	}

	for step := 1; step <= maxSteps; step++ { // [6] 最大步数
		resp, err := callWithRetry(ctx, client, input, toolParams)
		if err != nil {
			if ctx.Err() != nil {
				fmt.Println("已超时中止:", ctx.Err())
			} else {
				fmt.Println("API 错误:", err)
			}
			return
		}

		// [4] 从 output 中解析 function_call（一次可能有 0/1/N 个）
		var calls []responses.ResponseFunctionToolCall
		for _, item := range resp.Output {
			if item.Type == "function_call" {
				calls = append(calls, item.AsFunctionCall())
			}
		}

		// 没有 tool call，说明模型给出最终回答
		if len(calls) == 0 {
			fmt.Println("最终回答:", resp.OutputText())
			return
		}

		for _, call := range calls {
			fmt.Printf("[step %d] tool call: %s(%s)\n", step, call.Name, call.Arguments)

			// [5] 执行工具；未知工具或执行出错都作为结果喂回，让模型自行纠正
			result := fmt.Sprintf("未知工具: %s", call.Name)
			if tool, ok := toolsByName[call.Name]; ok {
				result = tool.run(json.RawMessage(call.Arguments))
			}
			fmt.Printf("[step %d] 工具结果: %s\n", step, result)

			// [5] 调用记录 + 执行结果一起喂回模型（无状态模式，call_id 必须对应）
			input = append(input, responses.ResponseInputItemUnionParam{
				OfFunctionCall: &responses.ResponseFunctionToolCallParam{
					CallID:    call.CallID,
					Name:      call.Name,
					Arguments: call.Arguments,
				},
			})
			input = append(input, responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: openai.String(call.CallID),
					Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
						OfString: openai.String(result),
					},
				},
			})
		}
	}
	fmt.Printf("已达到最大步数 %d，强制中止（模型可能陷入循环）\n", maxSteps)
}

// [6] 对 429/5xx 这类瞬时错误重试一次
func callWithRetry(ctx context.Context, client openai.Client, input []responses.ResponseInputItemUnionParam, tools []responses.ToolUnionParam) (*responses.Response, error) {
	for attempt := 1; attempt <= 2; attempt++ {
		resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
			Model: model,
			Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
			Tools: tools,
			Store: openai.Bool(false),
		})
		if err == nil {
			return resp, nil
		}
		var apiErr *openai.Error
		if attempt < 2 && errors.As(err, &apiErr) && (apiErr.StatusCode == 429 || apiErr.StatusCode >= 500) {
			fmt.Printf("瞬时错误(HTTP %d)，2 秒后重试\n", apiErr.StatusCode)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(2 * time.Second):
			}
			continue
		}
		return nil, err
	}
	return nil, errors.New("unreachable")
}
