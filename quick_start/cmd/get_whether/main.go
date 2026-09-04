package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

// 本地真正执行的函数：实际项目中可替换为调用天气 API、查数据库等
func getWeather(location string) string {
	return fmt.Sprintf("Sunny, 18°C, humidity 40%% (mock data for %s)", location)
}

func main() {
	client := openai.NewClient(
		// OpenCode Zen 网关：SDK 会自动拼接 /responses
		option.WithBaseURL("https://opencode.ai/zen/go/v1"),
		// key 从环境变量读取，不要硬编码
		option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)

	// chat.ChatMainFunc(&client)

	parameters := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{"type": "string", "description": "City and country e.g. Bogotá, Colombia"},
		},
		"required":             []string{"location"},
		"additionalProperties": false,
	}
	tool := responses.ToolParamOfFunction("get_weather", parameters, true)

	question := "What's the weather like in Paris today?"

	// 无状态多轮：每轮把完整上下文（用户消息 + 历史 function_call + 结果）重新传入
	input := []responses.ResponseInputItemUnionParam{
		{OfMessage: &responses.EasyInputMessageParam{
			Role:    responses.EasyInputMessageRoleUser,
			Content: responses.EasyInputMessageContentUnionParam{OfString: openai.String(question)},
		}},
	}

	// Agent 循环：模型可能连续多轮调用工具，直到给出最终回答
	for round := 1; round <= 10; round++ {
		var calls []responses.ResponseFunctionToolCall

		stream := client.Responses.NewStreaming(context.Background(), responses.ResponseNewParams{
			Model: "grok-4.6",
			Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
			Tools: []responses.ToolUnionParam{tool},
			// 关闭服务端会话存储，不依赖 previous_response_id（兼容第三方模型网关）
			Store: openai.Bool(false),
		})
		for stream.Next() {
			switch ev := stream.Current(); ev.Type {
			case "response.output_text.delta":
				// 最终回答的增量文本，边生成边打印
				fmt.Print(ev.AsResponseOutputTextDelta().Delta)
			case "response.output_item.done":
				// 输出项完成时，若是函数调用则收集起来
				item := ev.AsResponseOutputItemDone().Item
				if item.Type == "function_call" {
					calls = append(calls, item.AsFunctionCall())
				}
			}
		}
		if err := stream.Err(); err != nil {
			panic(err)
		}

		// 本轮没有函数调用 → 模型已给出最终回答（文本已在上面流式打印）
		if len(calls) == 0 {
			fmt.Println()
			return
		}

		// 逐个执行函数调用，把调用记录和执行结果都追加回 input
		for _, call := range calls {
			fmt.Printf("\n[第%d轮] 模型请求调用: %s(%s)\n", round, call.Name, call.Arguments)

			var args struct {
				Location string `json:"location"`
			}
			var result string
			switch {
			case call.Name != "get_weather":
				result = "unknown tool: " + call.Name
			case json.Unmarshal([]byte(call.Arguments), &args) != nil:
				result = "invalid arguments: " + call.Arguments
			default:
				result = getWeather(args.Location)
			}

			// 1) 回传函数调用本身（无状态模式下必须，否则模型不知道自己发起过什么调用）
			input = append(input, responses.ResponseInputItemUnionParam{
				OfFunctionCall: &responses.ResponseFunctionToolCallParam{
					CallID:    call.CallID,
					Name:      call.Name,
					Arguments: call.Arguments,
				},
			})
			// 2) 回传函数执行结果，call_id 必须对应
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
}
