package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"inventory-excercise/tools"
	"os"
	"time"

	"github.com/openai/openai-go/v3/option"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
)

const (
	model        = "deepseek-v4-flash"
	retryTimes   = 2
	maxSteps     = 5
	agentTimeout = 2 * time.Minute
)

var retryDelay = 2 * time.Second

// ErrMaxSteps is returned when the model still wants to call tools after the
// configured number of agent iterations has been exhausted.
var ErrMaxSteps = errors.New("agent exceeded max steps")

// responseCaller is one Responses API request. Keeping it as a function makes
// the retry, timeout, and max-step behavior testable without network access.
type responseCaller func(
	ctx context.Context,
	input responses.ResponseInputParam,
	agentTools []responses.ToolUnionParam,
) (*responses.Response, error)

func main() {
	client := openai.NewClient(
		// OpenCode Zen 网关：SDK 会自动拼接 /responses
		// option.WithBaseURL("https://opencode.ai/zen/go/v1"),
		// option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		option.WithBaseURL("https://api.deepseek.com/v1"),
		option.WithAPIKey(os.Getenv("DEEPSEEK_API_KEY")),
		// key 从环境变量读取，不要硬编码

	)
	ctx, cancel := context.WithTimeout(context.Background(), agentTimeout)
	defer cancel()

	fmt.Println("=== no-tool chat ===")
	if _, err := chat(ctx, &client, "用一句话说明你完成盘点任务的思路"); err != nil {
		fmt.Fprintf(os.Stderr, "chat failed: %v\n", err)
		os.Exit(1)
	}

	if err := agent_loop(ctx, clientResponseCaller(client)); err != nil {
		fmt.Fprintf(os.Stderr, "agent failed: %v\n", err)
		os.Exit(1)
	}
}

func chat(ctx context.Context, client *openai.Client, userMsg string) (string, error) {
	resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: model,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(userMsg),
		},
	})
	if err != nil {
		return "", err
	}
	fmt.Printf("chat output: %s\n", resp.OutputText())
	return resp.OutputText(), nil // 注意：OutputText 是方法，要加 ()
}

func clientResponseCaller(client openai.Client) responseCaller {
	return func(ctx context.Context, input responses.ResponseInputParam, agentTools []responses.ToolUnionParam) (*responses.Response, error) {
		return client.Responses.New(ctx, responses.ResponseNewParams{
			Model: model,
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: input,
			},
			Tools: agentTools,
		})
	}
}

func callWithRetry(ctx context.Context, callAPI responseCaller, input responses.ResponseInputParam, agentTools []responses.ToolUnionParam) (*responses.Response, error) {
	for try := 1; try <= retryTimes; try++ {
		resp, err := callAPI(ctx, input, agentTools)
		if err == nil {
			return resp, nil
		}

		// Do not spend the remaining timeout on another attempt if the caller
		// has already canceled or the total deadline has expired.
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		var apiErr *openai.Error
		if !errors.As(err, &apiErr) || (apiErr.StatusCode != 429 && apiErr.StatusCode < 500) {
			return nil, fmt.Errorf("responses API request failed: %w", err)
		}

		// retryTimes is 2, so a transient response is attempted at most twice:
		// the original request plus one retry.
		if try == retryTimes {
			return nil, fmt.Errorf("responses API request failed after retry (HTTP %d): %w", apiErr.StatusCode, err)
		}

		fmt.Printf("transient API error (HTTP %d); retrying once in %s\n", apiErr.StatusCode, retryDelay)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(retryDelay):
		}
	}

	// Unreachable when retryTimes is greater than zero.
	return nil, errors.New("no responses API attempt was made")
}

func agent_loop(ctx context.Context, callAPI responseCaller) error {
	agentTools := tools.BuildTools()
	systemPrompt := fmt.Sprintf(
		"你是一个只负责仓库库存盘点的机器人，禁止越界操作（尤其不要读取/搜索团队成员 team.txt 等无关文件）。\n"+
			"本轮任务总共只能进行不超过 %d 轮工具调用，必须尽早完成，因此请尽量在一个助手轮次里并行调用多个无依赖的工具。参考高效路径：\n"+
			"① search(\"库存\") 定位库存文件路径；\n"+
			"② 并行调用 read_file(① 返回的路径)、list_files()：内置知识库里的库存路径已过时、通常读取会失败（file not found），所以要在同一轮里用 list_files 拿到 data/ 下真实文件名并确认，避免反复猜错路径浪费轮次；\n"+
			"③ read_file 读取真正的库存清单（data/inventory.txt）;\n"+
			"④ calculator 汇总总件数与商品种数：文件内每个商品都要传入，数量为“待盘点”的商品数量填 0，但仍计入商品种数；\n"+
			"⑤ write_report 把最终报告以 JSON 字符串保存到 data/report.json，禁止只在文本里给报告。",
		maxSteps,
	)
	input := responses.ResponseInputParam{
		responses.ResponseInputItemParamOfMessage(
			systemPrompt,
			responses.EasyInputMessageRoleSystem,
		),
	}

	input = append(input, responses.ResponseInputItemParamOfMessage(
		"帮我执行盘点库存的操作，将盘点报告以 JSON 格式保存到 data/report.json。"+
			"报告必须包含以下字段：warehouse（string）、total_items（integer）、total_quantity（integer）、"+
			"unchecked_items（string 数组）、is_complete（boolean，"+
			"仅当所有商品都已有盘点数字时为 true，存在未盘点商品时必须为 false）。"+
			"保存报告必须调用 write_report 工具，禁止只在回复文本中给出报告。",
		responses.EasyInputMessageRoleUser,
	))

	for step := 1; step <= maxSteps; step++ {
		fmt.Printf("=== agent step %d/%d ===\n", step, maxSteps)
		resp, err := callWithRetry(ctx, callAPI, input, agentTools)
		if err != nil {
			return fmt.Errorf("step %d: %w", step, err)
		}

		// First collect every function call in this response, then execute all
		// of them. This preserves support for parallel tool calls.
		type toolCall struct {
			item   responses.ResponseOutputItemUnion
			result string
			failed bool
		}
		var calls []toolCall
		for _, item := range resp.Output {
			if item.Type != "function_call" {
				continue
			}

			fmt.Printf("→ call [%s] [%s] args=%s\n", item.CallID, item.Name, item.Arguments.OfString)
			result, err := tools.Dispatch(item.Name, json.RawMessage(item.Arguments.OfString))
			failed := err != nil
			if failed {
				// Tool failures are deliberately sent back as output strings
				// so the model can recover on the next turn.
				result = err.Error()
			}
			calls = append(calls, toolCall{item: item, result: result, failed: failed})
		}

		// A response without function calls is the normal final turn.
		if len(calls) == 0 {
			fmt.Println(resp.OutputText())
			return nil
		}

		for _, c := range calls {
			status := "ok"
			if c.failed {
				status = "error"
			}
			fmt.Printf("← result [%s] [%s] [%s] %s\n", c.item.CallID, c.item.Name, status, c.result)
		}

		// Stateless API: echo assistant items first, then append every output
		// after the corresponding calls while matching CallID values.
		for _, item := range resp.Output {
			switch item.Type {
			case "message":
				m := item.AsMessage()
				p := m.ToParam()
				input = append(input, responses.ResponseInputItemUnionParam{OfOutputMessage: &p})
			case "reasoning":
				if ri, ok := reasoningEcho(item); ok {
					input = append(input, ri)
				}
			case "function_call":
				input = append(input, responses.ResponseInputItemUnionParam{
					OfFunctionCall: &responses.ResponseFunctionToolCallParam{
						Name: item.Name, Arguments: item.Arguments.OfString, CallID: item.CallID,
					},
				})
			}
		}
		for _, c := range calls {
			input = append(input, responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: openai.String(c.item.CallID),
					Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
						OfString: openai.String(c.result),
					},
				},
			})
		}
	}

	// If execution reaches here, every allowed step contained at least one tool
	// call and the model never produced a final no-tool response.
	return fmt.Errorf("%w: stopped after %d steps", ErrMaxSteps, maxSteps)
}

// reasoningEcho 把输出的 reasoning item 转成输入项：DeepSeek 只认明文 content，
// 不支持 summary / encrypted_content，因此只回传 id + 明文 reasoning_text。
func reasoningEcho(item responses.ResponseOutputItemUnion) (responses.ResponseInputItemUnionParam, bool) {
	r := item.AsReasoning()
	content := make([]responses.ResponseReasoningItemContentParam, 0, len(r.Content))
	for _, part := range r.Content {
		if part.Text == "" {
			continue
		}
		content = append(content, responses.ResponseReasoningItemContentParam{
			Text: part.Text,
			Type: part.Type, // reasoning_text
		})
	}
	if len(content) == 0 {
		return responses.ResponseInputItemUnionParam{}, false
	}
	return responses.ResponseInputItemUnionParam{
		OfReasoning: &responses.ResponseReasoningItemParam{
			ID:      r.ID,
			Content: content,
		},
	}, true
}
