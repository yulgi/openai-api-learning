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
	model      = "deepseek-v4-flash"
	reTryTimes = 2
	maxLoop    = 5
)

func main() {
	client := openai.NewClient(
		// OpenCode Zen 网关：SDK 会自动拼接 /responses
		// option.WithBaseURL("https://opencode.ai/zen/go/v1"),
		// option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		option.WithBaseURL("https://api.deepseek.com/v1"),
		option.WithAPIKey(os.Getenv("DEEPSEEK_API_KEY")),
		// key 从环境变量读取，不要硬编码

	)
	ctx := context.Background()
	// chat(ctx, &client, "hello， 介绍一下你自己")

	agent_loop(ctx, client)

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

func callWithRetry(ctx context.Context, client openai.Client, input responses.ResponseInputParam, tools []responses.ToolUnionParam) (resp *responses.Response, err error) {
	for try := 0; try < reTryTimes; try++ {
		resp, err = client.Responses.New(ctx, responses.ResponseNewParams{
			Model: model,
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: input,
			},
			Tools: tools,
		})

		if err == nil {
			return
		}

		var apiErr *openai.Error
		if try < reTryTimes-1 && errors.As(err, &apiErr) && (apiErr.StatusCode == 429 || apiErr.StatusCode >= 500) {
			// 如果是api错误的话，就尝试重试
			fmt.Printf("瞬时错误(HTTP %d)，2 秒后重试\n", apiErr.StatusCode)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(2 * time.Second):
			}
			continue
		}
		return
	}
	return
}

func agent_loop(ctx context.Context, client openai.Client) {
	agentTools := tools.BuildTools()
	input := responses.ResponseInputParam{
		responses.ResponseInputItemParamOfMessage(
			"你是一个只负责仓库库存盘点的机器人，禁止越界操作（尤其不要读取/搜索团队成员 team.txt 等无关文件）。\n"+
				"本轮任务总共只能进行不超过 5 轮工具调用，必须尽早完成，因此请尽量在一个助手轮次里并行调用多个无依赖的工具。参考高效路径：\n"+
				"① search(\"库存\") 定位库存文件路径；\n"+
				"② 并行调用 read_file(① 返回的路径)、list_files()：内置知识库里的库存路径已过时、通常读取会失败（file not found），所以要在同一轮里用 list_files 拿到 data/ 下真实文件名并确认，避免反复猜错路径浪费轮次；\n"+
				"③ read_file 读取真正的库存清单（data/inventory.txt）;\n"+
				"④ calculator 汇总总件数与商品种数：文件内每个商品都要传入，数量为“待盘点”的商品数量填 0，但仍计入商品种数；\n"+
				"⑤ write_report 把最终报告以 JSON 字符串保存到 data/report.json，禁止只在文本里给报告。",
			responses.EasyInputMessageRoleSystem),
	}

	input = append(input, responses.ResponseInputItemParamOfMessage(
		"帮我执行盘点库存的操作，将盘点报告以 JSON 格式保存到 data/report.json。"+
			"报告必须包含以下字段：warehouse（string）、total_items（integer）、total_quantity（integer）、"+
			"unchecked_items（string 数组）、is_complete（boolean，"+
			"仅当所有商品都已有盘点数字时为 true，存在未盘点商品时必须为 false）。"+
			"保存报告必须调用 write_report 工具，禁止只在回复文本中给出报告。",
		responses.EasyInputMessageRoleUser,
	))

	for step := 0; step < maxLoop; step++ {
		resp, err := callWithRetry(ctx, client, input, agentTools)
		if err != nil {
			panic(err)
		}

		// 先执行本轮全部工具调用，拿到结果后再统一回传
		type toolCall struct {
			item   responses.ResponseOutputItemUnion
			result string
		}
		var calls []toolCall
		for _, item := range resp.Output {
			fmt.Println(item.Type)
			if item.Type != "function_call" {
				continue
			}
			result, err := tools.Dispatch(item.Name, json.RawMessage(item.Arguments.OfString))
			if err != nil {
				result = err.Error() // 错误作为字符串喂回，模型自愈
			}
			calls = append(calls, toolCall{item: item, result: result})
		}

		// 本轮一次工具调用都没有 → 模型给出最终文本，收尾退出
		if len(calls) == 0 {
			fmt.Println(resp.OutputText())
			return
		}

		for _, c := range calls {
			fmt.Printf("← result [%s] [%s] %s\n", c.item.CallID, c.item.Name, c.result)
		}

		// 无状态网关：把本轮 assistant 输出按原始顺序回传。
		// DeepSeek 校验：thinking mode + tools 时 reasoning_text 必须传回。
		// 关键坑：并行 tool call（一轮多个 function_call）时，function_call_output
		// 不能穿插在 function_call 之间，否则会打断 reasoning 与相邻 assistant 消息的
		// 邻接，触发 400 "reasoning_text must be passed back"。
		// 因此：先集中回传 reasoning / message / 全部 function_call，
		// 最后再统一追加全部 function_call_output。
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
			// 回传执行结果，CallID 必须与上面的 function_call 对应
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
