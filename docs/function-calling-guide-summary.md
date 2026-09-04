# OpenAI Function Calling 指南总结

> 来源：[Function calling | OpenAI API Docs](https://developers.openai.com/api/docs/guides/function-calling?api-mode=responses)
> 场景：Responses API 模式

---

## 一、核心概念

| 术语 | 含义 |
|------|------|
| **Tools / Function（工具）** | 我们告诉模型"你可以用"的功能。如 `get_weather(location)`、查账户、退款等 |
| **Tool call（工具调用）** | 模型发现需要某个工具才能回答时，返回的一种特殊响应。如对"巴黎天气如何？"返回 `get_weather(location="Paris")` |
| **Tool call output（工具输出）** | 你的应用执行工具后产生的结果，**必须带上 `call_id` 引用是哪次调用** |

**Functions 与 Tools 的关系**：

- **function 工具**：用 JSON Schema 定义参数（本文重点）
- **custom 工具**：自由文本输入输出（见第八节）
- **内置工具**：OpenAI 平台自带（web search、code interpreter、MCP 等）

## 二、调用流程（5 步闭环）

```
1. 发起请求（prompt + tools 定义）
        ↓
2. 模型返回 tool call（可能 0/1/N 个）
        ↓
3. 应用本地执行函数
        ↓
4. 带 tool output 发起第二次请求
        ↓
5. 收到最终回答（或更多 tool calls → 回到第 2 步循环）
```

Responses API 可以无限循环这个流程。这个编排循环如果不想自己写，可以用 Agents SDK。

## 三、定义函数

在每次请求的 `tools` 参数中声明，字段：

| 字段 | 说明 |
|------|------|
| `type` | 固定为 `"function"` |
| `name` | 函数名，如 `get_weather` |
| `description` | 何时、如何使用该函数——**写得越清楚，模型调用越准** |
| `parameters` | JSON Schema，定义输入参数 |
| `strict` | 是否强制严格模式（见第六节） |

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current temperature for a given location.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City and country e.g. Bogotá, Colombia"
      }
    },
    "required": ["location"],
    "additionalProperties": false
  },
  "strict": true
}
```

**Namespaces（命名空间）**：工具很多时可以按领域分组（`crm`、`billing`、`shipping`），帮助模型在服务不同系统的同名工具间做选择。

**Tool search（工具搜索）**：工具生态很大时，可以延迟加载部分工具，模型需要时先搜索再加载进上下文。仅 `gpt-5.4` 及之后模型支持。

## 四、Responses API 完整示例（Python）

```python
import json
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "name": "get_horoscope",
    "description": "Get today's horoscope for an astrological sign.",
    "parameters": {
        "type": "object",
        "properties": {
            "sign": {
                "type": "string",
                "description": "An astrological sign like Taurus or Aquarius",
            },
        },
        "required": ["sign"],
        "additionalProperties": false,
    },
}]

def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."

input_list = [{"role": "user", "content": "What is my horoscope? I am an Aquarius."}]

response = client.responses.create(
    model="gpt-5.6",
    tools=tools,
    input=input_list,
)

# 关键：把模型全部输出（含 reasoning、function_call 项）加回 input
input_list += response.output

for item in response.output:
    if item.type == "function_call":
        if item.name == "get_horoscope":
            sign = json.loads(item.arguments)["sign"]   # arguments 是 JSON 字符串
            horoscope = get_horoscope(sign)
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,                 # call_id 必须对应
                "output": horoscope,
            })

response = client.responses.create(
    model="gpt-5.6",
    tools=tools,
    input=input_list,
)
print(response.output_text)
```

## 五、处理函数调用的要点

1. **假设有多个调用**：一次响应可能包含 0、1 或 N 个 tool call，务必循环处理
2. **`arguments` 是 JSON 编码的字符串**，需要 `json.loads()` / `JSON.parse()` 后再用
3. **结果格式随意**：`function_call_output` 的 `output` 通常传字符串，JSON、错误码、纯文本都行，模型能自行解读
4. **推理模型（GPT-5 / o4-mini）特别注意**：响应中带 tool calls 的 **reasoning items 必须原样回传**，否则 API 报错。所以示例里 `input_list += response.output` 是把全部输出项（含 reasoning）都加回去

## 六、进阶配置

### Tool choice（控制模型何时调用工具）

| 值 | 行为 |
|----|------|
| `"auto"`（默认） | 0 个、1 个或多个，模型自己决定 |
| `"required"` | 必须至少调用一个 |
| `{"type": "function", "name": "get_weather"}` | 强制调用指定函数 |
| `{"type": "allowed_tools", "mode": "auto", "tools": [...]}` | 限制在子集中调用（不改动 tools 列表，**可保留 prompt caching 收益**） |
| `"none"` | 禁用，等同不传 tools |

### Parallel function calling（并行调用）

- 模型可能在单轮里并行调用多个函数（GPT-5 起支持，内置工具不能混入并行批次）
- 设 `parallel_tool_calls: false` 可强制每轮 0 或 1 个调用
- 微调模型单轮多调用时 strict mode 会被禁用

### Strict mode（严格模式）

`strict: true` 保证函数调用**严格遵循 schema**，而不是尽力而为。**官方建议始终开启**。

底层依赖 Structured Outputs，schema 有硬性要求：

1. 每个对象的 `additionalProperties` 必须为 `false`
2. `properties` 中**所有字段必须列入 `required`**——可选字段用 `"type": ["string", "null"]` 表达

**省略 `strict` 时的默认行为（易考点）**：

- **Responses API**：会尝试把 schema 自动规范化为 strict；无法规范化时回退 best-effort，此时响应里的工具显示 `strict: false`
- **Chat Completions API**：默认非严格
- 想在 Responses 里明确关闭严格模式，需显式设 `strict: false`

## 七、Streaming（流式工具调用）

设 `stream: true` 后，可以实时看到模型调用了哪个函数、参数如何逐字生成。

Responses API 的关键事件序列：

```
response.output_item.added            ← 每个函数调用一项（含 response_id、output_index、item）
response.function_call_arguments.delta ← 参数增量片段（delta）
response.function_call_arguments.done ← 参数完整生成完毕（含完整调用）
```

Chat Completions 则是在 chunk 的 `delta.tool_calls` 中接收增量。

## 八、Custom tools（自定义工具）

与 function 工具类似，但**不定义输入 schema——模型直接回传任意字符串**作为工具输入。适用场景：

- 不想把输入包成 JSON（如直接传一段代码）
- 用 CFG（上下文无关文法）约束模型输出格式

```json
{
  "type": "custom",
  "name": "code_exec",
  "description": "Executes arbitrary Python code."
}
```

模型返回的是 `custom_tool_call` 类型，`input` 是纯文本：

```json
{
  "type": "custom_tool_call",
  "call_id": "call_aGiF...",
  "name": "code_exec",
  "input": "print(\"hello world\")"
}
```

### Context-free grammars（CFG 文法约束）

用 `format` 参数给 custom 工具加文法，支持两种语法：**`lark`** 和 **`regex`**，可把模型输入约束为合法格式（如数学表达式、固定模式）：

```json
{
  "type": "custom",
  "name": "math_exp",
  "format": {
    "type": "grammar",
    "syntax": "lark",
    "definition": "start: expr\nexpr: term (SP ADD SP term)* ..."
  }
}
```

## 九、结合自己实践的坑（Go SDK + Zen 网关 + grok-4.6）

> 这是实际踩坑记录，文档里没有：

1. **`custom` 工具是 OpenAI 官方端点专属**——第三方模型（xAI grok 等）的兼容接口不认这个类型，直接 422 `Upstream request failed`。第三方模型只用 `type: "function"`
2. **无状态模式（`store: false`）下必须手动回传 `function_call` 项本身**，否则模型不知道自己发起过什么调用
3. Go SDK（openai-go/v3）中 `ResponseInputItemFunctionCallOutputParam.CallID` 是 `param.Opt[string]` 类型，要用 `openai.String()` 包装
4. `code_interpreter`（托管代码沙箱）同样仅限官方端点；第三方模型要"执行代码"就自己用 function 工具 + 本地执行（注意沙箱安全）

## 十、快速复习清单

```
□ 5 步闭环：tools 定义 → tool call → 本地执行 → 回传 output → 最终回答
□ arguments 是 JSON 字符串，要解析
□ call_id 必须一一对应
□ 一次可能有 N 个调用，循环处理
□ 推理模型的 reasoning items 必须回传
□ strict: true 要求 additionalProperties: false + 全字段 required
□ Responses 省略 strict 会自动规范化；Chat Completions 默认非严格
□ tool_choice: auto / required / 指定函数 / allowed_tools / none
□ parallel_tool_calls: false 禁止并行
□ 流式事件：output_item.added → arguments.delta → arguments.done
□ custom 工具：自由文本 + CFG（lark/regex），仅官方端点
□ 第三方模型网关：只用 function 工具，无状态需回传 function_call 项
```
