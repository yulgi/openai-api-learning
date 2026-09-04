package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

// 本地真正执行 Python 代码。
// 注意：直接 exec 不受信任的代码有安全风险，生产环境应加沙箱隔离（容器/超时/限权）
func runPython(code string) string {
	out, err := exec.Command("python3", "-c", code).CombinedOutput()
	if err != nil {
		return fmt.Sprintf("exit error: %v\n%s", err, out)
	}
	return string(out)
}

func main() {
	client := openai.NewClient(
		// OpenCode Zen 网关：SDK 会自动拼接 /responses
		option.WithBaseURL("https://opencode.ai/zen/go/v1"),
		// key 从环境变量读取，不要硬编码
		option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)

	response, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
		Model: "grok-4.6",
		Tools: []responses.ToolUnionParam{
			responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch),
		},
		Input: responses.ResponseNewParamsInputUnion{OfString: openai.String("What was a positive news story from today?")},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(response.OutputText())
}
