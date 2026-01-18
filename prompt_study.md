化与大模型应用学习指南

## 目录
1. [Agent 认知架构](#agent-认知架构)
2. [主流框架介绍](#主流框架介绍)
3. [ReAct 模式详解](#react-模式详解)
4. [LLM API 集成](#llm-api-集成)
5. [Prompt Engineering 优化技巧](#prompt-engineering-优化技巧)
6. [学习资源汇总](#学习资源汇总)
7. [学习步骤建议](#学习步骤建议)

---

## Agent 认知架构

### 核心概念

AI Agent（智能体）是能够感知环境、进行推理、采取行动并从反馈中学习的自主系统。现代 LLM-based Agent 的认知架构通常包含三个核心组件：**Planning（规划）**、**Memory（记忆）** 和 **Tool Use（工具使用）**。

### 为什么需要 Agent？

想象一下传统 LLM 与 Agent 的区别：
- **传统 LLM**：一问一答模式，无法执行多步骤任务，没有持久记忆，无法与外部系统交互
- **AI Agent**：能够分解复杂任务、记住上下文、调用工具完成实际操作，实现端到端的任务自动化

### Planning（规划）

规划是 Agent 将复杂任务分解为可执行子任务的能力。

#### 1. 任务分解（Task Decomposition）

```
用户请求："帮我分析这份销售报告并生成可视化图表"

Agent 规划过程：
├── 子任务1：读取并解析销售报告文件
├── 子任务2：提取关键数据指标
├── 子任务3：进行数据分析（趋势、对比）
├── 子任务4：选择合适的图表类型
└── 子任务5：生成可视化图表并保存
```

#### 2. 推理链（Chain of Thought）

Agent 通过逐步推理来解决问题：

```
问题：计算公司上季度的增长率

推理过程：
1. 需要获取上季度和上上季度的销售数据
2. 调用数据查询工具获取数据
3. 计算增长率 = (本期 - 上期) / 上期 × 100%
4. 格式化输出结果
```

#### 3. 目标导向规划

- **前向规划**：从当前状态出发，逐步推进到目标状态
- **后向规划**：从目标状态反推需要的前置条件
- **层次规划**：高层目标分解为低层可执行动作

### Memory（记忆）

记忆系统使 Agent 能够存储、检索和利用信息。

#### 1. 短期记忆（Short-term Memory）

```
作用：维护当前对话的上下文
实现：LLM 的 Context Window
特点：容量有限（如 128K tokens），会话结束后消失

示例：
用户：我叫张三
Agent：你好张三！（短期记忆保存了用户名字）
用户：我的名字是什么？
Agent：你的名字是张三。（从短期记忆中检索）
```

#### 2. 长期记忆（Long-term Memory）

```
作用：持久化存储跨会话的信息
实现：外部数据库 + 向量检索
特点：容量大，持久保存，支持语义检索

常用技术栈：
├── 向量数据库：Pinecone, Weaviate, Milvus, Chroma
├── 嵌入模型：OpenAI Embedding, BGE, E5
└── 检索策略：相似度检索, MMR, 混合检索
```

#### 3. 记忆架构示意

```
┌─────────────────────────────────────────────────┐
│                    Agent                        │
├─────────────────────────────────────────────────┤
│  ┌─────────────┐      ┌─────────────────────┐   │
│  │ 短期记忆     │      │ 长期记忆             │   │
│  │ (Context)   │ ←──→ │ (Vector DB)         │   │
│  │ 当前对话    │      │ 历史知识/用户画像    │   │
│  └─────────────┘      └─────────────────────┘   │
│           ↓                      ↓              │
│  ┌─────────────────────────────────────────┐    │
│  │           工作记忆 (Working Memory)       │    │
│  │     合并短期和长期记忆用于当前推理         │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Tool Use（工具使用）

工具使用能力让 Agent 能够与外部世界交互。

#### 1. 函数调用（Function Calling）

现代 LLM 支持结构化的函数调用：

```python
# 定义工具
tools = [
    {
        "name": "search_web",
        "description": "搜索互联网获取最新信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "execute_code",
        "description": "执行 Python 代码进行计算",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python 代码"}
            },
            "required": ["code"]
        }
    }
]

# Agent 决策过程
用户输入 → LLM 分析 → 选择工具 → 执行工具 → 获取结果 → 继续推理或返回
```

#### 2. 常见工具类型

| 工具类型 | 示例 | 用途 |
|---------|------|-----|
| 搜索工具 | Google Search, Bing | 获取实时信息 |
| 代码执行 | Python Interpreter | 数学计算、数据处理 |
| 文件操作 | 读写文件、解析文档 | 处理用户文件 |
| API 调用 | 天气、地图、数据库 | 获取外部数据 |
| 浏览器 | Playwright, Selenium | 网页交互 |

#### 3. 工具调用流程

```
1. 接收用户请求
   ↓
2. LLM 分析是否需要工具
   ├── 不需要 → 直接生成回答
   └── 需要 → 继续
   ↓
3. 选择合适的工具和参数
   ↓
4. 执行工具调用
   ↓
5. 获取工具返回结果
   ↓
6. 将结果整合到回答中
   ↓
7. 判断是否完成任务
   ├── 未完成 → 返回步骤2
   └── 完成 → 返回最终答案
```

---

## 主流框架介绍

### LangChain

LangChain 是最流行的 LLM 应用开发框架，提供了模块化的组件设计。

#### 核心概念

```
┌─────────────────────────────────────────────────────────────┐
│                      LangChain 架构                          │
├─────────────────────────────────────────────────────────────┤
│  Models        │ LLM 封装（OpenAI, Claude, Llama...）        │
├────────────────┼────────────────────────────────────────────┤
│  Prompts       │ 提示词模板、Few-shot 示例管理               │
├────────────────┼────────────────────────────────────────────┤
│  Chains        │ 组合多个组件的调用链                        │
├────────────────┼────────────────────────────────────────────┤
│  Agents        │ 动态决策使用哪些工具                        │
├────────────────┼────────────────────────────────────────────┤
│  Memory        │ 对话历史和上下文管理                        │
├────────────────┼────────────────────────────────────────────┤
│  Retrievers    │ 向量检索和文档加载                          │
└────────────────┴────────────────────────────────────────────┘
```

#### 代码示例

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate

# 1. 初始化模型
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 2. 定义工具
tools = [
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="用于数学计算"
    )
]

# 3. 创建提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 4. 创建 Agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 5. 执行
result = agent_executor.invoke({"input": "计算 123 * 456"})
```

#### LangGraph：状态机编排

LangGraph 是 LangChain 的扩展，用于构建复杂的多步骤工作流：

```python
from langgraph.graph import StateGraph, END

# 定义状态
class AgentState(TypedDict):
    messages: list
    next_step: str

# 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("analyze", analyze_task)
workflow.add_node("execute", execute_action)
workflow.add_node("review", review_result)

# 添加边
workflow.add_edge("analyze", "execute")
workflow.add_conditional_edges(
    "execute",
    should_continue,
    {"continue": "analyze", "end": END}
)

# 编译并运行
app = workflow.compile()
```

### Semantic Kernel

Semantic Kernel 是微软开源的 AI 编排框架，强调企业级应用开发。

#### 核心概念

```
┌─────────────────────────────────────────────────────────────┐
│                   Semantic Kernel 架构                       │
├─────────────────────────────────────────────────────────────┤
│  Kernel         │ 核心编排引擎，管理 AI 服务和插件           │
├─────────────────┼────────────────────────────────────────────┤
│  Plugins        │ 功能模块（原 Skills），包含多个函数         │
├─────────────────┼────────────────────────────────────────────┤
│  Functions      │ 原生函数或语义函数（Prompt 定义）          │
├─────────────────┼────────────────────────────────────────────┤
│  Connectors     │ AI 服务连接器（OpenAI, Azure, Hugging Face）│
├─────────────────┼────────────────────────────────────────────┤
│  Memory         │ 语义记忆，支持向量存储                      │
├─────────────────┼────────────────────────────────────────────┤
│  Planners       │ 自动规划执行步骤                           │
└─────────────────┴────────────────────────────────────────────┘
```

#### 代码示例（C#）

```csharp
using Microsoft.SemanticKernel;

// 1. 创建 Kernel
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion("gpt-4", apiKey);
var kernel = builder.Build();

// 2. 定义语义函数（使用 Prompt）
var summarizeFunction = kernel.CreateFunctionFromPrompt(
    "请用一句话总结以下内容：{{$input}}"
);

// 3. 定义原生函数
public class TimePlugin
{
    [KernelFunction]
    public string GetCurrentTime() => DateTime.Now.ToString();
}

kernel.ImportPluginFromType<TimePlugin>();

// 4. 执行
var result = await kernel.InvokeAsync(summarizeFunction,
    new() { ["input"] = "长文本内容..." });
```

#### 代码示例（Python）

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# 1. 创建 Kernel
kernel = sk.Kernel()

# 2. 添加 AI 服务
kernel.add_service(OpenAIChatCompletion(
    service_id="chat",
    ai_model_id="gpt-4",
    api_key="your-api-key"
))

# 3. 定义插件
class SearchPlugin:
    @sk.kernel_function
    def search_web(self, query: str) -> str:
        return f"搜索结果: {query}"

kernel.add_plugin(SearchPlugin(), "search")

# 4. 使用 Planner 自动规划
from semantic_kernel.planners import SequentialPlanner
planner = SequentialPlanner(kernel)
plan = await planner.create_plan("搜索最新的 AI 新闻")
result = await plan.invoke(kernel)
```

### 框架对比

| 特性 | LangChain | Semantic Kernel |
|-----|-----------|-----------------|
| 语言支持 | Python, JavaScript | C#, Python, Java |
| 设计理念 | 组件化、灵活 | 企业级、类型安全 |
| 学习曲线 | 较陡（概念多） | 中等 |
| 生态系统 | 非常丰富 | 微软生态集成好 |
| 适用场景 | 快速原型、复杂 Agent | 企业应用、.NET 项目 |
| 文档质量 | 完善 | 完善 |

### 选型建议

```
选择 LangChain：
├── Python 为主要开发语言
├── 需要快速原型验证
├── 需要丰富的第三方集成
└── 构建复杂的 RAG 或 Agent 系统

选择 Semantic Kernel：
├── .NET/C# 为主要开发语言
├── 需要与微软生态（Azure, M365）集成
├── 强调类型安全和企业级特性
└── 需要与现有 .NET 应用集成
```

---

## ReAct 模式详解

### 什么是 ReAct？

ReAct（Reasoning + Acting）是一种让 LLM 交替进行推理和行动的 Agent 设计模式。它通过**思考-行动-观察**的循环来解决复杂任务。

### 核心原理

```
传统方式：
输入 → LLM 直接输出答案（可能不准确）

ReAct 方式：
输入 → 思考 → 行动 → 观察 → 思考 → 行动 → 观察 → ... → 最终答案
```

### 思考-行动-观察循环

#### 循环流程图

```
┌────────────────────────────────────────────────────────────┐
│                       ReAct 循环                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│    ┌──────────┐                                            │
│    │  用户输入 │                                            │
│    └────┬─────┘                                            │
│         ↓                                                  │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐         │
│    │  Thought │ ──→ │  Action  │ ──→ │Observation│         │
│    │  (思考)   │     │  (行动)   │     │  (观察)   │         │
│    └──────────┘     └──────────┘     └────┬─────┘         │
│         ↑                                  │               │
│         └──────────────────────────────────┘               │
│                    (循环直到完成)                           │
│         ↓                                                  │
│    ┌──────────┐                                            │
│    │ 最终答案  │                                            │
│    └──────────┘                                            │
└────────────────────────────────────────────────────────────┘
```

#### 各步骤详解

1. **Thought（思考）**：LLM 分析当前情况，决定下一步做什么
2. **Action（行动）**：执行具体操作（调用工具、搜索、计算等）
3. **Observation（观察）**：获取行动的结果，作为下一轮思考的输入

### 实际示例

```
用户问题：北京今天的天气怎么样？适合户外活动吗？

=== 第一轮 ===
Thought: 用户想知道北京今天的天气和是否适合户外活动。
         我需要先查询北京的实时天气数据。
Action: search_weather(city="北京")
Observation: 北京今日天气：晴，气温 18-26°C，空气质量良，PM2.5: 45

=== 第二轮 ===
Thought: 我已经获得了天气数据。天气晴朗，温度适宜，空气质量良好。
         这些条件都表明适合户外活动。我可以给出最终答案了。
Action: finish(answer="北京今天天气晴朗，气温18-26°C，空气质量良好，
                       非常适合户外活动！建议您多喝水，做好防晒。")
Observation: [任务完成]

最终答案：北京今天天气晴朗，气温18-26°C，空气质量良好，
         非常适合户外活动！建议您多喝水，做好防晒。
```

### ReAct 提示词模板

```
你是一个有帮助的助手，可以使用以下工具来回答问题：

可用工具：
{tools_description}

请按照以下格式回答：

Question: 用户的问题
Thought: 分析问题，思考应该怎么做
Action: 工具名称[参数]
Observation: 工具返回的结果
... (重复 Thought/Action/Observation 直到得到答案)
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答

开始！

Question: {user_question}
```

### 代码实现

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}

    def run(self, question: str, max_iterations: int = 10) -> str:
        prompt = self._build_prompt(question)
        history = ""

        for i in range(max_iterations):
            # 获取 LLM 响应
            response = self.llm.generate(prompt + history)

            # 解析思考和行动
            thought, action = self._parse_response(response)
            history += f"\nThought: {thought}\nAction: {action}"

            # 检查是否完成
            if action.startswith("finish"):
                return self._extract_answer(action)

            # 执行行动并获取观察结果
            observation = self._execute_action(action)
            history += f"\nObservation: {observation}"

        return "达到最大迭代次数，未能得出结论"

    def _execute_action(self, action: str) -> str:
        tool_name, params = self._parse_action(action)
        if tool_name in self.tools:
            return self.tools[tool_name].execute(params)
        return f"未知工具: {tool_name}"
```

### 工作流设计实践

#### 复杂任务的 ReAct 工作流

```
任务：分析竞争对手的产品定价并生成报告

工作流设计：
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 信息收集                                             │
│   Thought: 需要收集竞争对手的产品价格信息                      │
│   Action: search_web("竞争对手A 产品定价 2024")               │
│   Action: search_web("竞争对手B 产品定价 2024")               │
├─────────────────────────────────────────────────────────────┤
│ Step 2: 数据整理                                             │
│   Thought: 需要将收集到的数据结构化整理                        │
│   Action: extract_pricing_data(raw_data)                    │
├─────────────────────────────────────────────────────────────┤
│ Step 3: 分析对比                                             │
│   Thought: 对比我们的产品与竞品的价格差异                      │
│   Action: compare_pricing(our_prices, competitor_prices)    │
├─────────────────────────────────────────────────────────────┤
│ Step 4: 生成报告                                             │
│   Thought: 根据分析结果生成可视化报告                         │
│   Action: generate_report(analysis_result)                  │
└─────────────────────────────────────────────────────────────┘
```

#### 设计原则

1. **明确的任务边界**：每个 Action 应该有清晰的输入输出
2. **错误处理**：考虑工具调用失败时的重试或降级策略
3. **循环限制**：设置最大迭代次数，避免无限循环
4. **上下文管理**：合理裁剪历史记录，避免超出 Token 限制

---

## LLM API 集成

### 主流模型 API 概览

| 模型 | 提供商 | 特点 | API 端点 |
|------|-------|------|----------|
| GPT-4/4o | OpenAI | 综合能力强，生态丰富 | api.openai.com |
| Claude 3.5 | Anthropic | 长上下文，安全性高 | api.anthropic.com |
| Llama 3 | Meta | 开源，可本地部署 | 自托管或云服务 |
| Gemini | Google | 多模态能力强 | generativelanguage.googleapis.com |
| 文心一言 | 百度 | 中文优化 | aip.baidubce.com |
| 通义千问 | 阿里 | 中文优化，开源版本 | dashscope.aliyuncs.com |

### API 调用模式

#### 1. OpenAI API 示例

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# 基础调用
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "你好！"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

#### 2. Anthropic Claude API 示例

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="你是一个有帮助的助手",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)

print(response.content[0].text)
```

#### 3. 统一接口封装

```python
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list, **kwargs) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: list, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, messages: list, **kwargs) -> str:
        # 转换消息格式
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=user_messages,
            **kwargs
        )
        return response.content[0].text
```

### 流式响应（Streaming）

流式响应可以提升用户体验，让用户更快看到输出：

```python
# OpenAI 流式响应
def stream_chat(prompt: str):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

# Anthropic 流式响应
def stream_chat_anthropic(prompt: str):
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
```

### 重试机制与错误处理

```python
import time
from functools import wraps

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    errors: tuple = (Exception,)
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"重试 {attempt + 1}/{max_retries}，等待 {delay}秒...")
                    time.sleep(delay)
                    delay *= exponential_base
        return wrapper
    return decorator

@retry_with_exponential_backoff(
    max_retries=3,
    errors=(openai.RateLimitError, openai.APIConnectionError)
)
def safe_chat(messages):
    return client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

### 成本优化策略

```
成本优化技巧：
├── 1. 选择合适的模型
│   └── 简单任务用 GPT-3.5/Claude Haiku，复杂任务用 GPT-4/Claude Sonnet
├── 2. 控制 Token 使用
│   ├── 精简 System Prompt
│   ├── 限制 max_tokens
│   └── 及时裁剪对话历史
├── 3. 缓存策略
│   ├── 缓存相同问题的回答
│   └── 使用 Embedding 做语义相似度判断
├── 4. 批量处理
│   └── 合并多个小请求为一个大请求
└── 5. 监控与预算
    ├── 设置 API 使用限额
    └── 监控 Token 使用趋势
```

---

## Prompt Engineering 优化技巧

### 提示词设计原则

#### 1. 清晰明确（Clarity）

```
❌ 不好的提示词：
"帮我写点东西"

✅ 好的提示词：
"请写一篇500字的产品介绍文案，介绍我们的智能手表产品。
目标受众：25-35岁的职场人士
突出卖点：健康监测、长续航、时尚外观
语气：专业但不失亲和力"
```

#### 2. 提供上下文（Context）

```
❌ 缺少上下文：
"这个代码有什么问题？"

✅ 提供充分上下文：
"以下是一段 Python 代码，用于从 API 获取用户数据并存入数据库。
运行时会出现 ConnectionTimeout 错误，请分析可能的原因并提供修复建议。

```python
def fetch_users():
    response = requests.get(API_URL)
    ...
```
```

#### 3. 指定输出格式（Format）

```
请分析以下文本的情感倾向，并以 JSON 格式返回结果：

{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0-1.0,
  "keywords": ["关键词1", "关键词2"]
}
```

#### 4. 分步骤引导（Step-by-step）

```
请按以下步骤分析这份财务报告：

1. 首先，总结报告的主要财务指标
2. 然后，分析同比和环比变化
3. 接着，识别潜在的风险点
4. 最后，给出改进建议

每个步骤请单独输出，使用标题标注。
```

### 常用技巧

#### 1. Few-shot Learning（少样本学习）

通过提供示例来引导模型输出格式和风格：

```
请将以下英文句子翻译成中文，保持原文的语气和风格。

示例：
English: The quick brown fox jumps over the lazy dog.
中文：敏捷的棕色狐狸跳过了懒惰的狗。

English: Actions speak louder than words.
中文：行动胜于言语。

现在请翻译：
English: Every cloud has a silver lining.
中文：
```

#### 2. Chain-of-Thought（思维链）

引导模型展示推理过程：

```
请解决以下数学问题，并展示详细的推理过程：

问题：一个商店进货价是80元，售价是120元，如果打8折销售，利润率是多少？

请按以下格式回答：
- 分析：[分析问题]
- 步骤1：[第一步计算]
- 步骤2：[第二步计算]
- ...
- 答案：[最终答案]
```

#### 3. Self-Consistency（自洽性）

多次采样并选择一致性最高的答案：

```python
def self_consistency_answer(question: str, n_samples: int = 5) -> str:
    answers = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": question}],
            temperature=0.7  # 增加多样性
        )
        answers.append(response.choices[0].message.content)

    # 投票选择最常见的答案
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

#### 4. Role Prompting（角色扮演）

```
你是一位拥有20年经验的资深软件架构师，专注于分布式系统设计。
你的回答应该：
- 考虑系统的可扩展性和高可用性
- 引用具体的技术方案和最佳实践
- 权衡不同方案的利弊
- 给出明确的建议

用户问题：如何设计一个日活100万用户的即时通讯系统？
```

### 高级模式

#### 1. System Prompt 设计

```python
system_prompt = """
你是一个专业的代码审查助手，具有以下特点：

## 职责
- 审查代码质量和最佳实践
- 发现潜在的 Bug 和安全问题
- 提供改进建议

## 审查维度
1. 代码风格：命名规范、格式化、注释
2. 逻辑正确性：边界条件、异常处理
3. 性能：时间复杂度、资源使用
4. 安全性：输入验证、敏感信息处理
5. 可维护性：代码结构、模块化程度

## 输出格式
使用以下结构输出审查结果：

### 问题列表
- [严重程度] 问题描述
  - 位置：代码位置
  - 建议：修复建议

### 总体评价
[对代码的整体评价和改进建议]
"""
```

#### 2. 结构化输出

使用 JSON Schema 约束输出格式：

```python
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "user", "content": "分析苹果公司的股票"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "stock_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "ticker": {"type": "string"},
                    "recommendation": {
                        "type": "string",
                        "enum": ["买入", "持有", "卖出"]
                    },
                    "reasons": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "risk_level": {
                        "type": "string",
                        "enum": ["低", "中", "高"]
                    }
                },
                "required": ["company", "ticker", "recommendation", "reasons"]
            }
        }
    }
)
```

#### 3. 提示词版本管理

```python
class PromptManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.prompts = {}

    def register(self, name: str, template: str, version: str = "1.0"):
        self.prompts[name] = {
            "template": template,
            "version": version,
            "created_at": datetime.now()
        }

    def get(self, name: str, **variables) -> str:
        template = self.prompts[name]["template"]
        return template.format(**variables)

    def ab_test(self, name_a: str, name_b: str, test_inputs: list) -> dict:
        """A/B 测试两个提示词版本"""
        results = {"a": [], "b": []}
        for input_data in test_inputs:
            results["a"].append(self._evaluate(name_a, input_data))
            results["b"].append(self._evaluate(name_b, input_data))
        return self._compare_results(results)
```

---

## 学习资源汇总

### 官方文档与教程

| 资源 | 链接 | 说明 |
|------|------|------|
| OpenAI 官方文档 | https://platform.openai.com/docs | API 参考、最佳实践 |
| Anthropic 文档 | https://docs.anthropic.com | Claude API、提示词指南 |
| LangChain 文档 | https://python.langchain.com/docs | 完整的框架教程 |
| Semantic Kernel | https://learn.microsoft.com/semantic-kernel | 微软官方教程 |
| LangGraph 教程 | https://langchain-ai.github.io/langgraph | 工作流编排指南 |
| Hugging Face | https://huggingface.co/docs | 开源模型使用 |

### 推荐书籍与课程

#### 书籍

1. **《Building LLM Apps》** - 全面介绍 LLM 应用开发
2. **《Prompt Engineering for Developers》** - 提示词工程实战
3. **《Designing Large Language Model Applications》** - 系统设计视角

#### 在线课程

1. **DeepLearning.AI - LangChain 课程系列**
   - LangChain for LLM Application Development
   - LangChain: Chat with Your Data
   - Functions, Tools and Agents with LangChain

2. **DeepLearning.AI - Prompt Engineering**
   - ChatGPT Prompt Engineering for Developers
   - Building Systems with the ChatGPT API

3. **Microsoft Learn - Semantic Kernel**
   - Build AI solutions with Semantic Kernel

### 开源项目参考

| 项目 | 描述 | 学习价值 |
|------|------|---------|
| AutoGPT | 自主任务执行 Agent | Agent 架构设计 |
| BabyAGI | 简化版任务规划 Agent | 任务分解与优先级 |
| MemGPT | 长期记忆管理 | Memory 系统设计 |
| AgentGPT | 网页版 Agent 平台 | 用户交互设计 |
| LangServe | LangChain 部署工具 | 生产环境部署 |
| Flowise | 可视化 LLM 工作流 | 无代码 Agent 构建 |
| Dify | 开源 LLM 应用平台 | 完整应用架构 |
| OpenDevin | 开源软件工程 Agent | 复杂 Agent 实现 |

### 技术博客与社区

1. **Lilian Weng's Blog** - 深度技术文章
2. **LangChain Blog** - 框架更新与案例
3. **Anthropic Research** - 安全与对齐研究
4. **Simon Willison's Weblog** - LLM 实践分享
5. **GitHub Trending** - 关注 AI Agent 相关项目

---

## 学习步骤建议

### 阶段一：基础入门（2-3 周）

```
目标：理解核心概念，能够调用 LLM API

学习内容：
├── 1. LLM 基础知识
│   ├── 了解 Transformer 架构基础
│   ├── 理解 Token、Context Window 概念
│   └── 掌握 Temperature、Top-p 等参数含义
│
├── 2. API 调用实践
│   ├── 注册 OpenAI/Anthropic 账号
│   ├── 完成基础的 API 调用
│   ├── 实现流式响应
│   └── 处理错误和重试
│
└── 3. Prompt Engineering 入门
    ├── 学习提示词设计原则
    ├── 练习 Few-shot 和 CoT
    └── 完成 3-5 个提示词优化练习

实践项目：
- 搭建一个简单的命令行聊天机器人
- 实现一个文本摘要/翻译工具
```

### 阶段二：框架应用（3-4 周）

```
目标：熟练使用 LangChain/Semantic Kernel 构建应用

学习内容：
├── 1. LangChain 核心组件
│   ├── Models、Prompts、Chains
│   ├── Memory 管理
│   ├── 工具定义与使用
│   └── Agent 创建与配置
│
├── 2. RAG（检索增强生成）
│   ├── 文档加载与分割
│   ├── 向量化与存储
│   ├── 检索策略优化
│   └── 完整 RAG Pipeline
│
└── 3. LangGraph 工作流
    ├── 状态图定义
    ├── 条件分支与循环
    └── 多 Agent 协作

实践项目：
- 构建企业知识库问答系统（RAG）
- 实现一个代码审查 Agent
```

### 阶段三：Agent 深入（4-5 周）

```
目标：设计和实现复杂的 Agent 系统

学习内容：
├── 1. Agent 认知架构
│   ├── 深入理解 Planning、Memory、Tool Use
│   ├── ReAct、Reflexion 等模式
│   └── 多 Agent 架构设计
│
├── 2. 高级工具集成
│   ├── 浏览器自动化
│   ├── 代码执行沙箱
│   ├── 数据库操作
│   └── 自定义工具开发
│
├── 3. 生产环境考虑
│   ├── 可观测性（Tracing、Logging）
│   ├── 安全性与权限控制
│   ├── 成本监控与优化
│   └── 评估与测试方法
│
└── 4. 前沿探索
    ├── Function Calling 最佳实践
    ├── 长上下文处理策略
    └── 多模态 Agent

实践项目：
- 开发一个自动化数据分析 Agent
- 构建一个软件开发辅助 Agent
- 设计多 Agent 协作系统
```

### 学习路线图

```
┌────────────────────────────────────────────────────────────────────┐
│                         AI Agent 学习路线                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Week 1-2         Week 3-5         Week 6-8         Week 9-12     │
│  ┌─────┐         ┌─────┐          ┌─────┐          ┌─────┐       │
│  │基础  │   →    │框架  │    →    │ RAG │    →    │Agent│       │
│  │概念  │         │入门  │          │实践  │          │深入 │       │
│  └─────┘         └─────┘          └─────┘          └─────┘       │
│     │               │                │                 │          │
│     ↓               ↓                ↓                 ↓          │
│  ┌─────┐         ┌─────┐          ┌─────┐          ┌─────┐       │
│  │API  │         │Lang │          │知识库│          │复杂 │       │
│  │调用  │         │Chain│          │问答  │          │Agent│       │
│  └─────┘         └─────┘          └─────┘          └─────┘       │
│     │               │                │                 │          │
│     ↓               ↓                ↓                 ↓          │
│  ┌─────┐         ┌─────┐          ┌─────┐          ┌─────┐       │
│  │Prompt│        │工具  │          │检索  │          │生产  │       │
│  │优化  │         │使用  │          │优化  │          │部署  │       │
│  └─────┘         └─────┘          └─────┘          └─────┘       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 实践项目建议

| 阶段 | 项目 | 技术栈 | 难度 |
|------|------|-------|------|
| 入门 | CLI 聊天机器人 | OpenAI API | ⭐ |
| 入门 | 多语言翻译工具 | Claude API + Prompt | ⭐ |
| 进阶 | 企业知识库问答 | LangChain + RAG | ⭐⭐ |
| 进阶 | 代码审查助手 | LangChain + Agent | ⭐⭐⭐ |
| 高级 | 数据分析 Agent | LangGraph + Tools | ⭐⭐⭐⭐ |
| 高级 | 多 Agent 协作系统 | LangGraph + Multi-Agent | ⭐⭐⭐⭐⭐ |

---

## 总结

### AI Agent 开发的核心要点

1. **认知架构三要素**：
   - Planning：任务分解与规划能力
   - Memory：短期与长期记忆管理
   - Tool Use：外部系统交互能力

2. **框架选择**：
   - LangChain：灵活、生态丰富，适合 Python 开发
   - Semantic Kernel：企业级、类型安全，适合 .NET 生态

3. **ReAct 模式**：
   - 思考-行动-观察循环
   - 适用于需要多步推理的复杂任务

### Prompt Engineering 的核心技巧

1. **基础原则**：清晰、提供上下文、指定格式、分步引导
2. **常用技巧**：Few-shot、Chain-of-Thought、Self-Consistency
3. **高级模式**：System Prompt 设计、结构化输出、版本管理

### 技术栈层次

```
应用层：Agent 应用、RAG 系统、工作流自动化
    ↓
编排层：LangChain / Semantic Kernel / LangGraph
    ↓
模型层：GPT-4 / Claude / Llama / 通义千问
    ↓
基础设施：API Gateway、向量数据库、监控系统
```

### 持续学习建议

1. **关注前沿发展**：Agent 技术迭代快，保持学习新模式
2. **动手实践**：通过项目巩固知识，积累经验
3. **社区参与**：参与开源项目，与同行交流
4. **评估思维**：建立评估体系，量化改进效果
