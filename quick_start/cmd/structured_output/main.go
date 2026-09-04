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

// 期望模型输出的结构：先定义 Go 结构体，JSON Schema 要与之完全对应
type UserProfile struct {
	Name  string `json:"name"`
	Age   int    `json:"age"`
	City  string `json:"city"`
	Job   string `json:"job"`
	Email string `json:"email"`
	// strict 模式下"可选字段"的写法：类型数组加 "null"，Go 侧用指针接收
	Phone *string `json:"phone"`
}

// 与 UserProfile 对应的 JSON Schema。
// strict: true 的硬性要求：additionalProperties 必须为 false，
// 且 properties 里所有字段都必须列入 required
var schema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"name":  map[string]any{"type": "string"},
		"age":   map[string]any{"type": "integer"},
		"city":  map[string]any{"type": "string"},
		"job":   map[string]any{"type": "string"},
		"email": map[string]any{"type": "string"},
		"phone": map[string]any{"type": []any{"string", "null"}},
	},
	"required":             []string{"name", "age", "city", "job", "email", "phone"},
	"additionalProperties": false,
}

func main() {
	client := openai.NewClient(
		// OpenCode Zen 网关：SDK 会自动拼接 /responses
		option.WithBaseURL("https://opencode.ai/zen/go/v1"),
		// key 从环境变量读取，不要硬编码
		option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)

	text := "张三，28岁，在北京做后端工程师，邮箱 zhangsan@example.com"

	response, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
		Model: "grok-4.6",
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String("从下面的文字中提取用户信息：" + text),
		},
		// Structured Outputs：通过 text.format 指定 json_schema，
		// 模型输出保证严格符合 schema（区别于只保证"是合法JSON"的旧 json_object 模式）
		Text: responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   "user_profile", // 仅限 a-z A-Z 0-9 下划线连字符，最长 64
					Schema: schema,
					Strict: openai.Bool(true), // 建议始终开启
				},
			},
		},
	})
	if err != nil {
		panic(err)
	}

	// 从 message 输出项里取出 JSON 字符串
	var raw string
	for _, item := range response.Output {
		if item.Type == "message" {
			for _, c := range item.AsMessage().Content {
				if c.Type == "output_text" {
					raw = c.Text
				}
			}
		}
	}
	fmt.Println("模型原始输出:", raw)

	// 反序列化成 Go 结构体，之后就是类型安全的字段访问
	var profile UserProfile
	if err := json.Unmarshal([]byte(raw), &profile); err != nil {
		panic(err)
	}
	fmt.Printf("解析结果: %+v\n", profile)
	fmt.Printf("姓名=%s 年龄=%d 城市=%s 职业=%s\n", profile.Name, profile.Age, profile.City, profile.Job)
	if profile.Phone != nil {
		fmt.Printf("电话=%s\n", *profile.Phone)
	} else {
		fmt.Println("电话：未提供（可选字段输出 null）")
	}
}
