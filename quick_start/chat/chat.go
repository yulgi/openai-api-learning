package chat

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
)

// ChatMainFunc 启动多轮对话循环
func ChatMainFunc(client *openai.Client) {

	// 对话历史：保存之前的问答记录，模型可据此理解上下文
	history := responses.ResponseInputParam{
		responses.ResponseInputItemParamOfMessage(
			"你是一个乐于助人的中文助手，请用中文简洁地回答问题。",
			responses.EasyInputMessageRoleSystem,
		),
	}

	fmt.Println("已进入多轮对话模式（输入 exit 或 quit 退出）：")
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("你: ")
		if !scanner.Scan() {
			break // EOF（Ctrl+D）
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "exit" || input == "quit" {
			fmt.Println("再见！")
			break
		}

		// 把本轮用户提问追加到历史
		history = append(history, responses.ResponseInputItemParamOfMessage(
			input, responses.EasyInputMessageRoleUser,
		))

		response, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
			Model: "grok-4.6",
			Input: responses.ResponseNewParamsInputUnion{OfInputItemList: history},
		})
		if err != nil {
			fmt.Println("请求出错:", err)
			// 回滚刚追加的提问，避免失败轮次污染历史
			history = history[:len(history)-1]
			continue
		}

		answer := response.OutputText()
		fmt.Println("助手:", answer)

		// 把回答也追加到历史，供后续轮次参考
		history = append(history, responses.ResponseInputItemParamOfMessage(
			answer, responses.EasyInputMessageRoleAssistant,
		))
	}
}
