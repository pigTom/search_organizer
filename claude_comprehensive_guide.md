# Comprehensive Guide to Claude Model and Claude Code (2026)

This guide provides a thorough overview of both Anthropic's Claude AI assistant and Claude Code CLI tool, covering their capabilities, usage, and best practices.

---

# Part 1: Claude Model - Anthropic's AI Assistant

## 1. Definition (什么是Claude)

Claude is a family of large language models (LLMs) developed by Anthropic, designed to be helpful, harmless, and honest. Claude is a conversational AI assistant capable of understanding and generating human-like text, analyzing images, writing code, and performing complex reasoning tasks. Named after mathematician Claude Shannon, the "father of information theory," Claude represents Anthropic's commitment to building safe and beneficial AI systems.

Claude is not a single model but rather a family of models with varying capabilities and performance characteristics, ranging from the ultra-fast Haiku to the powerful frontier-class Opus. All current Claude models support text and image input, text output, multilingual capabilities, and vision processing.

## 2. Core Problem Being Solved (解决的核心问题)

Claude addresses several fundamental challenges in the AI landscape:

**AI Safety and Alignment**: Traditional AI models often struggled with generating harmful, biased, or misleading content. Claude was built from the ground up with Constitutional AI (CAI), a technique that trains models to be helpful, harmless, and honest by having them critique and revise their own responses according to a set of principles.

**Context Understanding**: Many AI assistants struggled with maintaining context over long conversations or processing large documents. Claude 4.x models offer context windows up to 200,000 tokens (approximately 150,000 words or 500 pages), with Opus 4.5 supporting up to 1 million tokens - enabling analysis of entire codebases, books, or legal documents in a single conversation.

**Reasoning and Accuracy**: Earlier AI models often produced confident-sounding but incorrect answers (hallucinations). Claude 4.x models incorporate advanced reasoning capabilities and are trained to express uncertainty when appropriate, significantly reducing hallucinations and improving reliability for enterprise use cases.

**Specialized Task Performance**: Organizations needed AI that could handle specific workflows like software development, legal analysis, healthcare documentation, and creative writing. Claude 4.x models excel at these specialized tasks while maintaining general intelligence.

## 3. Core Technologies and Architecture (核心技术和架构)

### Constitutional AI (CAI)

Claude's defining feature is Constitutional AI, a training methodology that uses AI feedback to make models more helpful, harmless, and honest. The process involves:

1. **Red teaming**: Claude generates responses to potentially harmful prompts
2. **Self-critique**: Claude evaluates its own responses against constitutional principles
3. **Revision**: Claude revises responses to better align with those principles
4. **Reinforcement Learning from AI Feedback (RLAIF)**: The model is trained on these improved responses

### Model Architecture

Claude is built on transformer architecture with several key enhancements:

- **Extended Context Windows**: Up to 200K tokens (Sonnet/Haiku) or 1M tokens (Opus 4.5)
- **Multimodal Capabilities**: Native support for text and vision inputs
- **Tool Use**: Ability to interact with external tools and APIs through function calling
- **Parallel Processing**: Claude 4.x models excel at parallel tool execution, firing off multiple operations simultaneously
- **Context Awareness**: Claude 4.5 models can track remaining context window throughout conversations

### Training Methodology

- **Supervised Fine-Tuning**: Trained on high-quality conversational data
- **Constitutional AI**: Self-improvement through principle-based critique
- **RLAIF**: Reinforcement learning guided by AI feedback rather than just human feedback
- **Safety Classifications**: Opus 4 achieved "Level 3" on Anthropic's safety scale

## 4. Main Application Scenarios (主要应用场景)

### Software Development
- **Web and mobile application development**: The leading use case at 10.4% of enterprise usage
- **Code review and debugging**: Analyzing codebases to find bugs and security issues
- **Documentation generation**: Automatically creating technical documentation
- **Example**: Schrödinger's CTO describes Claude as turning ideas into code "in minutes instead of hours"

### Healthcare and Life Sciences
- **Clinical workflows**: Qualifying prior authorization requests, supporting claims appeals
- **Patient care coordination**: Triaging patients and identifying those needing urgent care
- **R&D acceleration**: Generating experimental protocols and supporting regulatory submissions
- **Medical record analysis**: Summarizing complex patient histories
- **Example**: Healthcare organizations like Banner Health and Elation Health use Claude to combat physician burnout

### Legal Services
- **Legal research automation**: Quickly finding relevant cases and precedents
- **Contract analysis**: Uploading and summarizing 100+ page contracts with key highlights
- **Document drafting**: Creating pleadings, demand letters, and legal briefs
- **Deposition preparation**: Simulating depositions and preparing questions

### Content Creation and Marketing
- **Scale content production**: Marketing teams report 4x content scaling while maintaining quality
- **Personalized campaigns**: Crafting targeted email campaigns and marketing materials
- **Academic writing**: Research assistance and academic paper writing
- **Example**: Claude holds 32% enterprise AI application market share with 30 million monthly active users

### Enterprise Operations
- **Data pipeline automation**: Matillion reduced pipeline creation from 40 hours to 1 hour
- **Business strategy optimization**: Analysis and strategic planning support
- **Career development**: Resume writing, interview preparation, and skill development
- **Document processing**: Analyzing and extracting information from large document sets

### Customer Support
- **Intelligent chatbots**: Providing accurate, context-aware customer service
- **Ticket triage**: Automatically categorizing and routing support requests
- **Knowledge base queries**: Answering complex questions from documentation

### Research and Analysis
- **Academic research**: Literature review and synthesis
- **Financial analysis**: Processing and analyzing financial documents
- **Market research**: Analyzing trends and generating insights

## 5. Mainstream Solutions and Implementations (主流解决方案)

### Claude Model Family (Current Generation - 4.x Series)

#### Claude Opus 4.5 (Latest - November 2025)
**Capabilities**: The flagship, frontier-intelligence model with 1 million token context window

**Strengths**:
- Best-in-class reasoning and complex problem solving
- Exceptional performance on multi-day software projects (completes in hours)
- Breakthrough in self-improving AI agents (peak performance in 4 iterations vs 10+ for competitors)
- Superior enterprise workflow management with better cross-file memory
- Excels at creating spreadsheets, slides, and documents

**Pricing**: $5 input / $25 output per million tokens

**Best for**: Enterprise R&D, complex coding projects, large-scale document analysis, agent workflows

#### Claude Sonnet 4.5 (September 2025)
**Capabilities**: The balanced choice with 200,000 token context window

**Strengths**:
- Optimal balance of intelligence, speed, and cost
- World's best coding model overall
- Exceptional frontend and UI development (pixel-perfect layouts)
- Strong performance in agentic tasks and computer use
- Industry-leading cybersecurity capabilities

**Pricing**: $3 input / $15 output per million tokens

**Weaknesses**: Slightly less capable than Opus for extreme complexity; smaller context window than Opus

**Best for**: Most production use cases, advanced Q&A, structured reporting, chat applications

#### Claude Haiku 4.5 (October 2025)
**Capabilities**: The speed-optimized model with 200,000 token context window

**Strengths**:
- Fastest response times in the Claude family (2x faster than Sonnet 4)
- 90% of Sonnet 4.5's performance at one-third the cost
- Safest model yet with lowest rate of misaligned behaviors
- Excellent for high-volume, low-latency applications

**Pricing**: $1 input / $5 output per million tokens

**Weaknesses**: Less capable for highly complex reasoning; not ideal for multi-step workflows requiring deep logic

**Best for**: Real-time chat, document review, UI scaffolding, high-volume processing

### Legacy Models (3.x Series)

The Claude 3 family (Opus 3, Sonnet 3.5, Haiku 3) and earlier versions (Claude 2.1, Claude 2, Claude 1) are still available but are being superseded by the 4.x series for most use cases.

### Deployment Options

#### Direct API Access
- **Anthropic API**: Direct access via api.anthropic.com
- **Official SDKs**: Python and TypeScript/JavaScript
- **REST API**: Full HTTP API for any programming language

#### Cloud Provider Integrations
- **Amazon Bedrock**: Fully managed service with enterprise features
- **Google Vertex AI**: Integration with Google Cloud Platform
- **Microsoft Azure AI Foundry**: Claude integrated into Azure ecosystem

#### Specialized Versions
- **Claude for Healthcare** (January 2026): HIPAA-ready infrastructure with medical database integrations (CMS, ICD-10, PubMed)
- **Claude Cowork** (January 2026): GUI version for non-technical users
- **Claude.ai**: Web-based chat interface (Pro, Max, Team, Enterprise tiers)

### Orchestration Strategy

**Multi-Model Approach**: Use Sonnet 4.5 as the orchestrator to break down complex problems, then deploy multiple Haiku 4.5 instances in parallel for subtasks, with Opus 4.5 as the careful reviewer for critical work.

## 6. Development History and Evolution (发展历程)

### 2023 - The Beginning

**March 2023**: Anthropic released Claude 1 and Claude Instant (lightweight version), initially available only to select users. The chatbot was named after mathematician Claude Shannon.

**July 2023**: Claude 2 launched with public availability, marking Claude's first widespread release.

**September 2023**: Amazon announced partnership with initial $1.25 billion investment, planning $4 billion total.

**October 2023**: Google invested $500 million with commitment for additional $1.5 billion.

**November-December 2023**: Claude 2.1 introduced 200K-token context window, targeting enterprise scenarios in legal, finance, and research.

### 2024 - Major Advancement

**March 2024**: Claude 3 family released (Haiku, Sonnet, Opus) on March 4, outperforming peers on most benchmarks. Amazon completed remaining $2.75 billion investment.

**June 2024**: Claude 3.5 Sonnet released (June 20), outperforming the larger Claude 3 Opus. Introduced Artifacts capability for real-time code preview in separate window.

**October 2024**: Upgraded "Claude 3.5 Sonnet (New)" and Claude 3.5 Haiku released (October 22). Public beta of "computer use" feature allowing desktop environment interaction.

**November 2024**: Amazon doubled investment with additional $4 billion.

### 2025 - Frontier Models and Agents

**February 2025**: Claude 3.7 Sonnet released (February 24) as pioneering hybrid AI reasoning model with choice between rapid responses and step-by-step reasoning. Claude Code CLI tool initially released.

**March 2025**: Google invested another $1 billion. Anthropic raised $3.5 billion Series E (Lightspeed Venture Partners leading), achieving $61.5 billion valuation.

**May 2025**: Claude Sonnet 4 and Claude Opus 4 released (May 22) with new API features (code execution tool, Model Context Protocol connector, Files API). Claude Code became generally available. Opus 4 classified as "Level 3" on Anthropic's safety scale.

**October 2025**: Claude Haiku 4.5 released. Google cloud partnership announced, providing access to 1 million custom TPUs with 1+ gigawatt AI compute capacity expected by 2026.

**November 2025**: Claude Opus 4.5 released. Nvidia and Microsoft partnership announced with up to $15 billion investment, Anthropic planning $30 billion computing capacity purchase from Azure.

### 2026 - Specialization and Accessibility

**January 2026**:
- Claude Cowork released with GUI for non-technical users
- Claude for Healthcare division launched with HIPAA infrastructure
- Claude Sonnet 4.5 released with enhanced capabilities
- "Labs" division introduced with Mike Krieger (former CPO) joining
- Claude reaches 30 million monthly active users

### Key Milestones Summary

- **2023**: Foundation and initial partnerships
- **2024**: Major model improvements and computer use capabilities
- **2025**: Frontier-class models, agentic capabilities, massive investments
- **2026**: Specialized solutions and broader accessibility

## 7. Learning Resources (学习资料)

### Official Documentation

- [Claude Platform Documentation](https://platform.claude.com/docs/en/home) - Complete API reference and guides
- [Models Overview](https://platform.claude.com/docs/en/about-claude/models/overview) - Detailed model specifications
- [Features Overview](https://docs.claude.com/en/api/overview) - API features and capabilities
- [Prompt Engineering Overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) - Official prompting guide
- [Claude API Platform](https://claude.com/platform/api) - Developer platform homepage

### Getting Started Guides

- [Get Started with Claude](https://platform.claude.com/docs/en/get-started) - Quick start tutorial
- [Anthropic Academy: Claude API Development Guide](https://www.anthropic.com/learn/build-with-claude) - Comprehensive API development course
- [Getting Started with Claude 4 API: Developer's Walkthrough](https://blog.logrocket.com/getting-started-claude-4-api-developers-walkthrough/) - Practical walkthrough

### Best Practices and Techniques

- [Prompting Best Practices - Official](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices) - Claude 4.x prompting guide
- [Prompt Engineering Best Practices Blog](https://claude.com/blog/best-practices-for-prompt-engineering) - In-depth techniques
- [Claude Prompt Engineering Best Practices (2026)](https://promptbuilder.cc/blog/claude-prompt-engineering-best-practices-2026) - 2026 checklist
- [Mastering Prompt Engineering for Claude](https://www.walturn.com/insights/mastering-prompt-engineering-for-claude) - Advanced strategies
- [AWS Prompt Engineering Techniques Guide](https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/) - AWS Bedrock integration

### Integration Tutorials

- [Claude API Integration Guide 2025: Complete Tutorial](https://collabnix.com/claude-api-integration-guide-2025-complete-developer-tutorial-with-code-examples/) - Code examples in multiple languages
- [Mastering Claude Tool API: A Deep Dive](https://sparkco.ai/blog/mastering-claude-tool-api-a-deep-dive-for-developers) - Advanced tool use
- [Claude API: Integration Guide & Best Practices](https://www.tokenmetrics.com/blog/claude-api-guide) - Production best practices
- [Anthropic Claude - Make Integration](https://apps.make.com/anthropic-claude) - No-code integration platform

### Model Comparisons and Analysis

- [Claude AI Models Compared: Opus 4.5, Sonnet 4.5 & Haiku 4.5](https://toolsdock.com/articles/claude-ai-models-comparison/) - Comprehensive comparison
- [Claude Haiku 4.5 vs Sonnet 4.5: Detailed Comparison](https://www.creolestudios.com/claude-haiku-4-5-vs-sonnet-4-5-comparison/) - Performance analysis
- [Anthropic Claude Models Complete Guide](https://www.codegpt.co/blog/anthropic-claude-models-complete-guide) - All models explained
- [Comparing Claude 4.5 Haiku and Sonnet for AWS](https://www.cloudthat.com/resources/blog/comparing-claude-45-haiku-and-sonnet-for-aws-ai-and-data-workloads) - AWS workload optimization

### Pricing and Planning

- [Anthropic Claude API Pricing 2026](https://www.metacto.com/blogs/anthropic-api-pricing-a-full-breakdown-of-costs-and-integration) - Complete cost breakdown
- [Claude API Pricing Guide 2026](https://www.aifreeapi.com/en/posts/claude-api-pricing-per-million-tokens) - Per-million-token costs
- [Claude All Models Available: Choosing the Right Tier](https://www.datastudios.org/post/claude-ai-all-models-available-opus-4-5-sonnet-4-5-haiku-4-5-3-series-legacy-and-how-to-choose) - Model selection guide

### Industry-Specific Resources

- [Claude for Lawyers: Best Use Cases & Prompts](https://rankings.io/blog/claude-for-lawyers/) - Legal profession guide
- [Claude for Digital Marketing in 2026](https://marketingagent.blog/2026/01/08/how-to-use-claude-for-digital-marketing-in-2026-complete-guide-with-case-studies-strategies/) - Marketing strategies
- [Claude in Life Sciences: Practical Applications](https://intuitionlabs.ai/articles/claude-code-life-science-applications) - Healthcare and research

### Community Resources

- [Claude Use Cases - Official](https://claude.com/resources/use-cases) - Real-world applications
- [Customer Stories](https://claude.com/customers) - Case studies from enterprises
- [Anthropic API Postman Collection](https://www.postman.com/postman/anthropic-apis/documentation/dhus72s/claude-api) - API testing tools

### News and Updates

- [Anthropic News](https://www.anthropic.com/news/claude-opus-4-5) - Official announcements
- [Claude Revenue and Usage Statistics (2026)](https://www.businessofapps.com/data/claude-statistics/) - Market insights

### Video and Interactive Resources

- [Claude Workbench](https://platform.claude.com) - Browser-based API testing (generate code from sessions)
- Interactive tutorials available through Anthropic Academy

### Books and Academic Resources

- [Claude (language model) - Wikipedia](https://en.wikipedia.org/wiki/Claude_(language_model)) - Comprehensive overview
- [Anthropic - Wikipedia](https://en.wikipedia.org/wiki/Anthropic) - Company history
- [Report: Anthropic Business Breakdown & Founding Story](https://research.contrary.com/company/anthropic) - In-depth company analysis

## 8. Common Questions and Issues (常见问题)

### Q1: Which Claude model should I choose for my use case?

**Answer**:
- **Choose Opus 4.5 ($5/$25 per million tokens)** for: Complex reasoning, large codebases requiring 1M token context, enterprise R&D, multi-day coding projects, agent orchestration where accuracy is critical
- **Choose Sonnet 4.5 ($3/$15 per million tokens)** for: Most production use cases, balanced performance and cost, frontend development, general coding tasks, structured reporting
- **Choose Haiku 4.5 ($1/$5 per million tokens)** for: High-volume applications, real-time chat, quick document processing, UI scaffolding, cost-sensitive deployments

**Multi-model strategy**: Use Sonnet to orchestrate, Haiku for parallel subtasks, Opus for final review.

### Q2: How do I handle rate limits and 429 errors?

**Answer**: Claude API rate limits vary by tier:
- **Tier 1**: 50 requests per minute (RPM)
- **Tier 4**: 4,000 RPM (80x increase)

**Solutions**:
1. **Implement exponential backoff**: Retry with increasing delays (e.g., 1s, 2s, 4s, 8s)
2. **Monitor response headers**: Track `x-ratelimit-remaining` and `x-ratelimit-reset` headers
3. **Use prompt caching**: Cached tokens don't count toward input token rate limits, potentially 5x throughput increase
4. **Request tier upgrade**: Higher usage tiers available through Anthropic support
5. **Use Message Batches API**: 50% cost reduction for non-urgent, large-volume requests
6. **Optimize request patterns**: Batch smaller requests, reduce unnecessary API calls

90% of rate limit issues resolve within 5 minutes. If persistent, check your tier limits and upgrade if needed.

### Q3: How can I reduce hallucinations and improve accuracy?

**Answer**:
1. **Give permission to express uncertainty**: Include "If you're unsure, say so rather than guessing" in your prompt
2. **Use structured outputs**: Request specific formats (JSON, markdown tables) to enforce consistency
3. **Provide clear context**: Include relevant background information and examples
4. **Break complex tasks into steps**: Use prompt chaining for multi-step reasoning
5. **Implement verification**: Add post-processing checks, secondary moderation, or rule-based filters
6. **Use extended thinking**: Claude 3.7 and later support "thinking mode" for step-by-step reasoning
7. **Request citations**: Ask Claude to reference specific parts of provided documents
8. **Test with few-shot examples**: Include 3-5 examples of desired output format

Claude 4.x models have significantly lower hallucination rates than previous generations, especially when given explicit permission to acknowledge uncertainty.

### Q4: What's the difference between Claude API and Claude.ai web interface?

**Answer**:
- **Claude.ai**: Web-based chat interface with Pro ($20/month), Max, Team, and Enterprise tiers
  - Best for: Interactive conversations, quick prototyping, non-technical users
  - Includes: Artifacts, Projects, Citations, web interface features
  - Limited: Cannot integrate into applications, no programmatic control

- **Claude API**: Programmatic access via REST API
  - Best for: Production applications, automation, custom integrations
  - Includes: Full control, streaming, batch processing, tool use, embeddings
  - Pricing: Pay-per-token usage (varies by model)
  - Requires: Technical implementation, API key management

Choose Claude.ai for human interaction, Claude API for building applications.

### Q5: How do I effectively use Claude's 200K+ token context window?

**Answer**:
**Best practices**:
1. **Structure large inputs**: Use clear headings, sections, and delimiters (e.g., XML tags)
2. **Place instructions strategically**: Critical instructions at both beginning and end
3. **Use document metadata**: Include table of contents or summaries for very long documents
4. **Monitor context usage**: Claude 4.5 models are context-aware and can manage their own limits
5. **Leverage prompt caching**: Cache static content (documentation, code context) to reduce costs
6. **Progressive disclosure**: For extremely long contexts, consider chunking with retrieval

**Context window sizes**:
- Haiku 4.5: 200,000 tokens (~150,000 words)
- Sonnet 4.5: 200,000 tokens
- Opus 4.5: 1,000,000 tokens (~750,000 words)

One token ≈ 0.75 words. 200K tokens ≈ 500 pages of text.

### Q6: How do I implement Claude's tool use / function calling?

**Answer**:
**Basic implementation**:
```python
import anthropic

client = anthropic.Anthropic(api_key="your_api_key")

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
}]

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}]
)
```

**Key features**:
- **Parallel tool use**: Claude 4.x can call multiple tools simultaneously
- **Tool chaining**: Claude can use tool results to inform subsequent tool calls
- **Computer use**: Claude 3.5+ can interact with desktop environments (beta)

**Best practices**:
- Provide clear, detailed tool descriptions
- Use structured JSON schemas for parameters
- Handle tool errors gracefully
- Validate tool outputs before returning to Claude

### Q7: What are the security best practices for using Claude API?

**Answer**:
**API Key Security**:
1. **Never hardcode API keys**: Use environment variables or secure vaults (AWS Secrets Manager, Azure Key Vault)
2. **Rotate keys periodically**: Minimize impact of potential leaks
3. **Use workspace-specific keys**: Segment API keys by use case through Anthropic workspaces
4. **Restrict key permissions**: Use least-privilege access when available

**Data Security**:
1. **Don't send sensitive data**: Avoid PII, credentials, or confidential information unless necessary
2. **Use Claude for Healthcare**: For HIPAA compliance, use the specialized healthcare version
3. **Implement data filtering**: Sanitize inputs before sending to API
4. **Review outputs**: Check responses for inadvertent data leakage

**Application Security**:
1. **Validate all inputs**: Prevent prompt injection attacks
2. **Implement rate limiting**: Protect against abuse and unexpected costs
3. **Log and monitor**: Track API usage, errors, and anomalies
4. **Use moderation**: Add content filtering for user-generated inputs

Anthropic does not train on API data by default (opt-in only for Trust & Safety).

### Q8: How do I optimize costs when using Claude API?

**Answer**:
**Cost optimization strategies**:
1. **Choose the right model**: Use Haiku for simpler tasks (5x cheaper than Opus)
2. **Implement prompt caching**: Cache static content (system prompts, documentation) - reduces costs by ~90% for cached portions
3. **Use Message Batches API**: 50% cost reduction for non-time-sensitive requests
4. **Optimize prompt length**: Be concise but clear; remove unnecessary examples
5. **Stream responses**: Stop generation early if you have the needed information
6. **Multi-model orchestration**: Use Haiku for subtasks, Sonnet for orchestration, Opus only when necessary
7. **Monitor token usage**: Track input/output tokens; optimize high-cost queries
8. **Set max_tokens appropriately**: Don't request more tokens than needed

**Example cost comparison (per million tokens)**:
- Haiku 4.5: $1 input / $5 output
- Sonnet 4.5: $3 input / $15 output (3x Haiku)
- Opus 4.5: $5 input / $25 output (5x Haiku)

For a 10,000-token input + 2,000-token output task:
- Haiku: $0.02
- Sonnet: $0.06
- Opus: $0.10

### Q9: What's the difference between Claude's streaming and non-streaming responses?

**Answer**:
**Non-streaming (default)**:
- Returns complete response after generation finishes
- Simpler to implement
- Best for: Batch processing, background jobs, automated workflows

**Streaming**:
- Returns response incrementally as tokens are generated
- Better user experience for interactive applications
- Enables early stopping if answer is found
- Best for: Chat interfaces, real-time applications, long responses

**Implementation example**:
```python
# Non-streaming
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(message.content)

# Streaming
with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain quantum computing"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

**Considerations**: Streaming requires maintaining connection state; harder to retry failed requests.

### Q10: How do I migrate from another LLM (like GPT-4) to Claude?

**Answer**:
**Key differences to address**:
1. **API structure**: Claude uses Messages API (different from OpenAI's Chat Completions)
2. **Prompting style**: Claude responds well to clear, direct instructions and XML tags for structure
3. **System prompts**: Claude treats system messages differently; test your existing prompts
4. **Token limits**: Claude offers larger context (200K-1M tokens vs typical 128K)
5. **Tool/function calling**: Similar concept but different implementation

**Migration steps**:
1. **Install Anthropic SDK**: `pip install anthropic` or `npm install @anthropic-ai/claude-code`
2. **Update API calls**: Adapt to Messages API format
3. **Test prompts**: Claude 4.x models prefer explicit instructions; may need rephrasing
4. **Leverage strengths**: Use Claude's extended context, vision, and tool use capabilities
5. **Optimize for model**: Claude excels at structured outputs, coding, and analysis
6. **Monitor performance**: Track quality, latency, and cost metrics
7. **Gradual rollout**: A/B test before full migration

**Resources**:
- Review prompt engineering guides specific to Claude
- Use Claude Workbench to test and generate code
- Join Anthropic community for migration support

Many organizations use multiple models; consider a multi-LLM strategy for optimal results.

---

# Part 2: Claude Code - The Official CLI Tool

## 1. Definition (什么是Claude Code)

Claude Code is Anthropic's official command-line interface (CLI) tool that brings agentic AI capabilities directly into your terminal and development environment. It is an autonomous coding assistant that understands your codebase, executes routine tasks, explains complex code, and handles complete development workflows - all through natural language commands.

Unlike traditional code completion tools, Claude Code acts as an intelligent agent that can independently read files, run commands, search the web, edit code, execute tests, and manage git workflows. It's designed to handle everything from quick code explanations to multi-day software projects, working alongside developers as a collaborative pair programmer.

Claude Code integrates natively with popular IDEs (VS Code, JetBrains) and can connect to hundreds of external tools through the Model Context Protocol (MCP), making it a comprehensive development automation platform rather than just a coding assistant.

## 2. Core Problem Being Solved (解决的核心问题)

Claude Code addresses several critical pain points in modern software development:

**Context Switching Overhead**: Developers constantly switch between their IDE, terminal, browser, documentation, and various tools. Claude Code consolidates these workflows into a single interface where you can ask questions, make changes, run tests, and manage git operations without leaving your development context.

**Repetitive Development Tasks**: Much of software development involves routine work like writing boilerplate, updating documentation, refactoring code, fixing linting errors, and writing tests. Claude Code automates these repetitive tasks, freeing developers to focus on creative problem-solving and architectural decisions.

**Codebase Understanding**: Understanding large, unfamiliar codebases is time-consuming and error-prone. Claude Code can analyze entire projects, explain complex code relationships, trace execution flows, and answer questions about code structure and behavior - dramatically reducing onboarding time and debugging effort.

**Incomplete Automation**: Existing tools require developers to manually orchestrate workflows (git add, commit, push, create PR). Claude Code handles end-to-end workflows autonomously, from reading GitHub issues to writing code, running tests, and submitting pull requests - all through natural language instructions.

**Tool Integration Complexity**: Developers use dozens of tools (JIRA, Slack, Sentry, databases, APIs) but integrating them requires significant effort. Claude Code's MCP support provides standardized connections to hundreds of services, giving AI direct access to your entire development ecosystem.

## 3. Core Technologies and Architecture (核心技术和架构)

### Agent Architecture

Claude Code is built on an agentic architecture where the AI autonomously plans and executes multi-step tasks:

1. **Task Understanding**: Interprets natural language instructions in development context
2. **Planning**: Breaks down complex requests into executable steps
3. **Tool Orchestration**: Selects and uses appropriate tools (file operations, terminal commands, web search)
4. **Execution**: Performs actions with proper error handling and retries
5. **Verification**: Tests changes, runs validation, ensures correctness
6. **Iteration**: Adapts plans based on results and feedback

### Built-in Tool System

Claude Code provides a comprehensive set of tools accessible to the AI agent:

**File Operations**:
- **Read**: View file contents with syntax awareness
- **Write**: Create new files with complete content
- **Edit**: Make precise string replacements in existing files
- **Glob**: Find files using pattern matching (e.g., `**/*.ts`)
- **Grep**: Search file contents with regex support

**Terminal Integration**:
- **Bash**: Execute shell commands with output capture
- **Background execution**: Run long-running processes asynchronously
- **Command chaining**: Execute dependent commands sequentially

**Code Understanding**:
- **Syntax-aware editing**: Context-sensitive code modifications
- **Multi-file operations**: Work across entire codebases simultaneously
- **Notebook support**: Read and edit Jupyter notebooks (.ipynb)

**Git Workflow**:
- **Commit creation**: Analyze changes and generate appropriate commit messages
- **Branch management**: Create, switch, and merge branches
- **PR creation**: Generate pull requests with summaries and test plans
- **GitHub integration**: Read issues, comment on PRs, manage workflows

**Web Capabilities**:
- **WebFetch**: Retrieve and analyze web content
- **WebSearch**: Search the web for current information
- **Documentation lookup**: Find and read API documentation

**Specialized Tools**:
- **NotebookEdit**: Modify Jupyter notebook cells
- **Task management**: Track multi-step tasks with TodoWrite
- **Skills**: Execute custom workflows and commands

### Model Context Protocol (MCP)

MCP is Claude Code's extension mechanism for connecting to external services:

**Architecture**:
- **Open standard**: Developed by Anthropic for AI-tool integrations
- **Server-based**: Each MCP server provides tools, resources, and prompts
- **Configuration**: Managed through `.mcp.json` file in your project
- **Lazy loading**: Claude Code 2.1+ loads MCP tools on-demand, reducing context usage by 95%

**Capabilities**:
- **Tools**: Functions the AI can call (e.g., create JIRA ticket, query database)
- **Resources**: Data sources the AI can access (e.g., file systems, APIs)
- **Prompts**: Reusable prompt templates and workflows

**Popular MCP Servers**:
- Development: GitHub, GitLab, Linear, JIRA
- Databases: PostgreSQL, MySQL, MongoDB, Redis
- Observability: Sentry, DataDog, New Relic
- Communication: Slack, Discord, Email
- Cloud: AWS, Google Cloud, Azure
- And hundreds more from the community

### Customization System

**Hooks**: Middleware for customizing Claude Code's behavior
- **PreToolUse**: Add context before tool execution
- **SessionStart**: Initialize based on agent type or project
- **Output**: JSON specifying skills to activate, additional context

**Skills**: Custom collections of capabilities
- **Slash commands**: Quick shortcuts (e.g., `/test`, `/deploy`)
- **Agents**: Purpose-built workflows (e.g., bug triage, PR review)
- **Bundled installations**: Install related tools together

**Plugins**: Comprehensive extension packages
- Combine slash commands, agents, MCP servers, and hooks
- Single-command installation
- Marketplace available with community contributions

### IDE Integration

**Native Extensions**:
- **VS Code**: Full-featured extension with inline editing
- **VS Code forks**: Cursor, Windsurf support
- **JetBrains**: IntelliJ, PyCharm, WebStorm integration

**Features**:
- Inline code suggestions with context
- Side-by-side diff view for changes
- Terminal integration within IDE
- Project context awareness

### Parallel Agent Execution

Claude Code 2.1+ supports running multiple agents simultaneously:
- Execute different tasks in parallel
- Useful for mobile/remote development
- Background task execution while working on other items

## 4. Main Application Scenarios (主要应用场景)

### Code Explanation and Learning
**Use case**: Understanding unfamiliar codebases or complex algorithms
**Examples**:
- "Explain how the authentication flow works in this codebase"
- "What does this recursive function do and what's the time complexity?"
- "Trace the execution path when a user clicks the submit button"

**Benefits**: Dramatically reduces onboarding time; helps developers learn new frameworks and patterns

### Feature Development
**Use case**: Building new features from requirements to implementation
**Examples**:
- "Add a dark mode toggle to the settings page"
- "Implement pagination for the user list with 20 items per page"
- "Create a REST API endpoint for uploading profile images"

**Workflow**: Claude Code reads requirements, designs implementation, writes code, creates tests, and runs validation

**Real example**: Schrödinger reports Claude Code turns ideas into code "in minutes instead of hours"

### Refactoring and Code Quality
**Use case**: Improving code structure without changing functionality
**Examples**:
- "Refactor this component to use React hooks instead of class components"
- "Extract duplicate logic from these three files into a shared utility"
- "Rename the function getCwd to getCurrentWorkingDirectory across the entire project"

**Benefits**: Automated refactoring is safer and faster than manual changes; maintains consistency

### Bug Fixing and Debugging
**Use case**: Identifying and resolving issues
**Examples**:
- "Find and fix the bug causing the login form to crash"
- "Debug why the database query is slow"
- "Identify security vulnerabilities in the authentication code"

**Capabilities**: Claude can analyze stack traces, reproduce issues, apply fixes, and verify solutions

### Testing and Quality Assurance
**Use case**: Writing and running tests
**Examples**:
- "Write unit tests for the UserService class"
- "Add integration tests for the checkout flow"
- "Run all tests and fix any failures"

**Workflow**: Claude generates test cases, implements tests, runs test suites, and resolves failures iteratively

### Documentation
**Use case**: Creating and maintaining documentation
**Examples**:
- "Add JSDoc comments to all exported functions"
- "Generate API documentation from this Express.js server"
- "Update the README with installation instructions"

**Benefits**: Keeps documentation synchronized with code; ensures comprehensive coverage

### Git Workflow Management
**Use case**: End-to-end version control operations
**Examples**:
- "Review my changes and create a git commit"
- "Create a pull request for this feature"
- "Read GitHub issue #123 and implement the requested fix"

**Capabilities**: Generates appropriate commit messages following repository conventions, creates descriptive PRs with test plans, manages branches

**Best practice**: Claude Code analyzes git history to match existing commit message style

### Build and Deployment
**Use case**: Running build processes and deployments
**Examples**:
- "Build the project and fix any compilation errors"
- "Deploy to staging environment"
- "Run the linter and fix all issues automatically"

**Workflow**: Executes commands, monitors output, resolves errors, retries as needed

### Database Operations (via MCP)
**Use case**: Querying and managing databases
**Examples**:
- "Show me all users who registered in the last week"
- "Create a migration to add an email_verified column"
- "Optimize this slow database query"

**Requirements**: MCP server for your database (PostgreSQL, MySQL, etc.)

### Integration Development (via MCP)
**Use case**: Working with external services
**Examples**:
- "Create a JIRA ticket for this bug with stack trace"
- "Send a Slack message to #engineering about the deployment"
- "Query Sentry for errors in the last 24 hours"

**Benefits**: Seamless integration with your entire development ecosystem

## 5. Mainstream Solutions and Implementations (主流解决方案)

### Installation Methods

#### Native Installer (Recommended - 2026)
**Platforms**: macOS, Linux, Windows

**Installation**:
```bash
# macOS/Linux
curl -fsSL https://claude.ai/install.sh | bash

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex
```

**Advantages**:
- No Node.js dependency
- No package manager conflicts
- Automatic background updates
- Recommended by Anthropic

**Version-specific installation**:
```bash
curl -fsSL https://claude.ai/install.sh | bash -s 2.0.22
```

#### NPM Installation (Legacy - Deprecated)
```bash
npm install -g @anthropic-ai/claude-code
```

**Note**: npm installation is deprecated; migrate to native installer

**Migration command**:
```bash
claude install
```

### Access Requirements

Claude Code is available with:
- **Claude Pro**: $20/month individual subscription
- **Claude Max**: Enhanced individual subscription
- **Claude Team**: Team subscription with premium seats
- **Claude Enterprise**: Enterprise subscription
- **Claude Console**: Developer console account

### Claude Code Versions

#### Version 2.1.x (January 2026 - Current)
**Major updates**: Versions 2.1.0 through 2.1.9 released in January 2026

**Key improvements**:
- 109 CLI refinements
- MCP Tool Search with lazy loading (95% context reduction)
- Enhanced hooks system with `agent_type` parameter
- Improved parallel agent execution
- Better session management

**Notable quote**: "The biggest productivity upgrade since the tool launched"

#### Version 2.0.x
- General availability release (May 2025)
- Stable API and command structure
- Full IDE integration support

### IDE Integrations

#### VS Code Extension
**Features**:
- Inline code editing with diff view
- Integrated terminal
- Project-wide context awareness
- Real-time collaboration with AI

**Installation**: Available via VS Code marketplace

#### VS Code Forks
**Supported**:
- Cursor: AI-first code editor
- Windsurf: Enhanced VS Code variant

**Compatibility**: Full Claude Code feature parity

#### JetBrains IDEs
**Supported IDEs**:
- IntelliJ IDEA
- PyCharm
- WebStorm
- And other JetBrains products

**Features**: Native plugin with JetBrains UI integration

### Claude Cowork (January 2026)

**Description**: GUI version of Claude Code for non-technical users

**Target audience**: Product managers, designers, analysts who need AI coding assistance without terminal expertise

**Features**:
- Graphical interface for all Claude Code capabilities
- Simplified workflow for common tasks
- File management without command line

### Mobile Development (2026)

**Capability**: On-the-go software development via smartphones connected to cloud VMs

**Use cases**:
- Code during commutes
- Quick fixes from mobile devices
- Review and approve AI-generated changes

**Features**: Multiple parallel agents for multitasking

### GitHub Actions Integration

**Purpose**: Automate Claude Code workflows in CI/CD

**Examples**:
- Scheduled code quality sweeps
- Automated documentation updates
- PR review automation
- Test generation on commit

### MCP Marketplace Ecosystem

**Community Platforms**:
- **Glama.ai**: MCP server directory with ratings
- **Awesome Claude Code**: Curated GitHub lists
- **Claude Fast**: Extensions and addons catalog

**Top MCP Servers (2026)**:
1. **GitHub MCP**: Issues, PRs, code review integration
2. **PostgreSQL MCP**: Database query and management
3. **Slack MCP**: Team communication
4. **Sentry MCP**: Error monitoring and debugging
5. **Linear MCP**: Issue tracking and project management
6. **AWS MCP**: Cloud resource management
7. **Filesystem MCP**: Extended file operations
8. **Browser MCP**: Web automation
9. **Memory MCP**: Persistent context across sessions
10. **Search MCP**: Enhanced web search capabilities

**Configuration example** (`.mcp.json`):
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "${DATABASE_URL}"
      }
    }
  }
}
```

### Agent SDK (Python & TypeScript)

**Purpose**: Build custom AI agents using Claude Code's capabilities

**Languages**: Python and TypeScript

**Installation**:
```bash
# Python
pip install anthropic

# TypeScript/JavaScript
npm install @anthropic-ai/claude-code
```

**Use cases**:
- Custom automation workflows
- Specialized domain agents (e.g., code review agent)
- Integration into existing applications
- Programmatic Claude Code control

**Resources**:
- [Agent SDK Overview](https://docs.anthropic.com/en/docs/claude-code/sdk)
- [Python SDK Reference](https://platform.claude.com/docs/en/agent-sdk/python)
- [Complete Guide to Building Agents](https://nader.substack.com/p/the-complete-guide-to-building-agents)

## 6. Development History and Evolution (发展历程)

### February 2025 - Initial Release

**Launch**: Claude Code initially released alongside Claude 3.7 Sonnet

**Initial capabilities**:
- Terminal-based AI coding assistant
- Basic file operations and git integration
- Natural language command interface
- Context-aware codebase understanding

**Reception**: Immediate adoption by early users; identified as "powerful accelerator" for development

### May 2025 - General Availability

**Milestone**: Claude Code became generally available alongside Claude 4 models

**Major features added**:
- IDE integrations (VS Code, JetBrains)
- Enhanced git workflow automation
- Improved multi-file editing
- Tool use expansion

**API Features**: Launched with code execution tool, Model Context Protocol connector, Files API

### October 2025 - Web Version

**Release**: Web-based version of Claude Code launched

**Purpose**: Browser-based access for users without terminal setup

**Capabilities**: Full Claude Code features through web interface

### November 2025 - Infrastructure Expansion

**Partnership**: Nvidia and Microsoft announced major partnership

**Investment**: Up to $15 billion in Anthropic; $30 billion computing capacity from Azure

**Impact**: Massive scaling of Claude Code backend infrastructure

### January 2026 - Version 2.1 Series (Major Update)

**Versions**: 2.1.0 through 2.1.9 released

**Date**: January 7-15, 2026

**Major improvements**:
- **MCP Tool Search**: Lazy loading reduces context usage by 95%
- **109 CLI refinements**: Described as "biggest productivity upgrade since launch"
- **Enhanced hooks**: SessionStart hook includes `agent_type` parameter
- **Parallel agents**: Multiple agents can run simultaneously
- **Mobile support**: Smartphone development via cloud VMs

**Cowork Launch**: GUI version for non-technical users (January 2026)

**Ecosystem Growth**: Hundreds of MCP servers now available through community marketplaces

### Key Milestones Summary

- **Feb 2025**: Initial beta release with core features
- **May 2025**: General availability and IDE integrations
- **Oct 2025**: Web version launch
- **Jan 2026**: Version 2.1 series with major productivity enhancements
- **Jan 2026**: Cowork GUI launch for broader accessibility

### Evolution Trends

**From**: Simple coding assistant
**To**: Comprehensive agentic development platform

**Expansion areas**:
1. **Tool ecosystem**: From built-in tools to hundreds of MCP integrations
2. **Interface options**: Terminal → IDE extensions → Web → GUI → Mobile
3. **Autonomy**: Assistance → Task execution → End-to-end workflows
4. **Customization**: Fixed behavior → Hooks → Skills → Plugins
5. **Deployment**: Developer machines → Cloud VMs → CI/CD integration

## 7. Learning Resources (学习资料)

### Official Documentation

- [Claude Code Documentation](https://code.claude.com/docs/en/overview) - Complete official documentation
- [Claude Code Setup Guide](https://code.claude.com/docs/en/setup) - Installation and configuration
- [CLI Reference](https://code.claude.com/docs/en/cli-reference) - Complete command reference
- [MCP Integration Guide](https://code.claude.com/docs/en/mcp) - Connect external tools
- [GitHub Repository](https://github.com/anthropics/claude-code) - Official open-source repository
- [Product Page](https://claude.com/product/claude-code) - Features and capabilities overview

### Getting Started Guides

- [ClaudeLog Installation Guide](https://claudelog.com/install-claude-code/) - Step-by-step installation
- [Installation & Prerequisites for Product Managers](https://ccforpms.com/getting-started/installation) - Non-technical setup guide
- [The Definitive Guide to Claude Code](https://blog.devgenius.io/the-definitive-guide-to-claude-code-from-first-install-to-production-workflows-6d37a6d33e40) - First install to production
- [Claude Code Installation Guide for Windows](https://claude.ai/public/artifacts/d5297b60-4c2c-4378-879b-31cc75abdc98) - Windows-specific guide
- [Claude Code for the Rest of Us](https://www.whytryai.com/p/claude-code-beginner-guide) - Beginner-friendly tutorial

### Best Practices and Advanced Guides

- [Claude Code Best Practices for Agentic Coding](https://www.anthropic.com/engineering/claude-code-best-practices) - Official best practices from Anthropic
- [CLAUDE.md: Best Practices from Prompt Learning](https://arize.com/blog/claude-md-best-practices-learned-from-optimizing-claude-code-with-prompt-learning/) - Advanced optimization techniques
- [Ultimate Guide to Claude Code Setup: Hooks, Skills & Actions](https://aibit.im/blog/post/ultimate-guide-to-claude-code-setup-hooks-skills-actions) - Comprehensive customization guide
- [How I Use Claude Code (+ My Best Tips)](https://www.builder.io/blog/claude-code) - Real-world workflows from Builder.io
- [I Spent Months Building the Ultimate Claude Code Setup](https://medium.com/@sattyamjain96/i-spent-months-building-the-ultimate-claude-code-setup-heres-what-actually-works-ba72d5e5c07f) - Tested setup recommendations

### Tutorials and Practical Examples

- [Claude Code: A Guide With Practical Examples (DataCamp)](https://www.datacamp.com/tutorial/claude-code) - Hands-on tutorial with examples
- [Cooking with Claude Code: The Complete Guide](https://www.siddharthbharath.com/claude-code-the-complete-guide/) - Comprehensive practical guide
- [Claude Code and What Comes Next (Ethan Mollick)](https://www.oneusefulthing.org/p/claude-code-and-what-comes-next) - Thoughtful analysis and use cases
- [Claude Code in Life Sciences: Practical Applications](https://intuitionlabs.ai/articles/claude-code-life-science-applications) - Domain-specific examples

### Video and Interactive Courses

- [Claude Code in Action (Anthropic Skilljar)](https://anthropic.skilljar.com/claude-code-in-action) - Official interactive course covering:
  - Tool use systems
  - Context management
  - Visual communication workflows
  - Custom automation
  - MCP server integration
- [Claude Code Course for Product Managers](https://ccforpms.com/) - Free interactive course teaching AI-powered product management

### Reference Resources

- [Claude Code CLI Cheatsheet](https://shipyard.build/blog/claude-code-cheat-sheet/) - Config, commands, prompts, best practices
- [20+ Most Important Claude Code CLI Tricks (2025/2026)](https://mlearning.substack.com/p/20-most-important-claude-code-tricks-2025-2026-cli-january-update) - Power user tips
- [Claude Code 2.1 NEW Features](https://mlearning.substack.com/p/claude-code-21-new-features-january-2026) - January 2026 update details
- [Claude Code Complete Troubleshooting Guide](https://smartscope.blog/en/generative-ai/claude/claude-code-troubleshooting-guide/) - Common issues and solutions

### MCP and Customization

- [Ultimate Guide to Claude MCP Servers & Setup (2026)](https://generect.com/blog/claude-mcp/) - Comprehensive MCP guide
- [Top 10 Essential MCP Servers for Claude Code](https://apidog.com/blog/top-10-mcp-servers-for-claude-code/) - Best MCP servers for developers
- [Customize Claude Code with Plugins](https://www.anthropic.com/news/claude-code-plugins) - Official plugin documentation
- [A Comprehensive Guide to the Claude Code SDK](https://apidog.com/blog/a-comprehensive-guide-to-the-claude-code-sdk/) - SDK deep dive
- [Claude Code Notification Hooks MCP](https://glama.ai/mcp/servers/@nkyy/claude-code-notify-mcp) - Notification integration example

### Community Resources

- [Awesome Claude Code (GitHub)](https://github.com/jmanhype/awesome-claude-code) - Curated list of plugins, MCP servers, integrations
- [Awesome Claude Code (hesreallyhim)](https://github.com/hesreallyhim/awesome-claude-code) - Skills, hooks, slash-commands, applications
- [Awesome Claude - Visual Directory](https://awesomeclaude.ai/awesome-claude-code) - Visual catalog of resources
- [The Ultimate Claude Code Resource List (2026 Edition)](https://www.scriptbyai.com/claude-code-resource-list/) - Comprehensive community resources
- [Claude Fast - Extensions & Addons](https://claudefa.st/blog/tools/mcp-extensions/best-addons) - MCP and extension directory

### Specialized Use Cases

- [Claude Cowork Comprehensive Guide (2026)](https://elephas.app/blog/claude-cowork-comprehensive-guide) - GUI version guide
- [Claude Code in Life Sciences](https://intuitionlabs.ai/articles/claude-code-life-science-applications) - Healthcare and research applications

### News and Updates

- [CHANGELOG](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md) - Official changelog
- [Releases](https://github.com/anthropics/claude-code/releases) - GitHub releases
- [Anthropic - Claude Code Release Notes](https://releasebot.io/updates/anthropic/claude-code) - Latest updates tracked
- [ClaudeLog](https://claudelog.com/) - Community news, docs, and best practices

### SDK and Integration

- [Agent SDK Overview](https://docs.anthropic.com/en/docs/claude-code/sdk) - Official SDK documentation
- [Agent SDK Reference - Python](https://platform.claude.com/docs/en/agent-sdk/python) - Python API reference
- [The Complete Guide to Building Agents with Claude SDK](https://nader.substack.com/p/the-complete-guide-to-building-agents) - Agent development tutorial
- [Claude Code as MCP Server](https://github.com/steipete/claude-code-mcp) - Use Claude Code as MCP server

### Chinese Resources

Many resources are available in English only, but the official documentation and community tutorials are increasingly translated. Check ClaudeLog and community forums for Chinese-language guides.

## 8. Common Questions and Issues (常见问题)

### Q1: How do I install and set up Claude Code?

**Answer**:
**Recommended method** (Native installer):
```bash
# macOS/Linux
curl -fsSL https://claude.ai/install.sh | bash

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex
```

**Post-installation verification**:
```bash
claude doctor
```

**Migration from npm**:
If you previously installed via npm, migrate to native installer:
```bash
claude install
```

**Requirements**:
- Claude Pro, Max, Team, Enterprise, or Console account
- Internet connection
- macOS, Linux, or Windows

**First use**:
1. Run `claude` in terminal
2. Sign in with your Claude account credentials
3. Navigate to your project directory
4. Start chatting: "Help me understand this codebase"

### Q2: Why am I getting Claude Code API errors?

**Answer**:
**Common causes**:
1. **Temporary connectivity issues**: 90% resolve within 5 minutes
2. **Server maintenance**: May take up to 1 hour during high traffic
3. **Network problems**: Check your internet connection
4. **Authentication issues**: Verify you're logged in (`claude doctor`)

**Solutions**:
1. **Restart Claude Code**: Often resolves temporary issues
2. **Check status**: Visit Anthropic status page for outages
3. **Verify credentials**: Run `claude logout` then `claude login`
4. **Update Claude Code**: Ensure you're on latest version
5. **Check firewall**: Ensure api.anthropic.com is accessible

**Not actual API outages**: Most "API errors" are client-side connectivity issues, not server problems

**If persistent**:
- Check network connectivity
- Verify API access with your subscription tier
- Contact Anthropic support with error details

### Q3: How do I customize Claude Code with hooks and skills?

**Answer**:
**Hooks** (middleware for behavior customization):

Create `.claude/hooks/` directory in your project:
```javascript
// .claude/hooks/preToolUse.js
export default function preToolUse({ tool, input }) {
  // Add context before tool execution
  return {
    additionalContext: "Follow project coding standards in CONTRIBUTING.md"
  };
}
```

```javascript
// .claude/hooks/sessionStart.js
export default function sessionStart({ agent_type }) {
  // Activate skills based on agent type
  if (agent_type === 'testing') {
    return { activate: ["testing-patterns", "jest-conventions"] };
  }
  return { activate: ["react-ui-patterns", "typescript-best-practices"] };
}
```

**Skills** (custom workflows):

Create `.claude/skills/` directory:
```yaml
# .claude/skills/test-and-commit.yaml
name: Test and Commit
description: Run tests and commit if they pass
steps:
  - Run test suite
  - If tests pass, create git commit
  - Generate conventional commit message
```

**Installation**:
Skills and hooks are automatically loaded from `.claude/` directory when Claude Code starts.

**Best practices**:
- Use PreToolUse hooks for project-specific context
- Use SessionStart hooks to activate relevant skills
- Keep hooks lightweight to avoid latency
- Version control your `.claude/` directory

### Q4: How do I connect Claude Code to external tools (MCP)?

**Answer**:
**Setup MCP servers**:

1. **Create `.mcp.json`** in your project root:
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "${DATABASE_URL}"
      }
    }
  }
}
```

2. **Set environment variables**:
```bash
export GITHUB_TOKEN="your_github_token"
export DATABASE_URL="postgresql://user:pass@localhost/db"
```

3. **Restart Claude Code**: MCP servers load automatically

**Usage**:
Once configured, Claude Code can access MCP tools naturally:
- "Create a GitHub issue for this bug"
- "Query the database for all users created today"
- "Send a Slack message about deployment status"

**Popular MCP servers**:
- **GitHub**: `@modelcontextprotocol/server-github`
- **PostgreSQL**: `@modelcontextprotocol/server-postgres`
- **Filesystem**: `@modelcontextprotocol/server-filesystem`
- **Slack**: `@slack/mcp-server`
- **Sentry**: `@sentry/mcp-server`

**Discovery**:
Browse available MCP servers at:
- Glama.ai MCP directory
- Awesome Claude Code GitHub repositories
- Claude Fast extensions catalog

**Context optimization** (Claude Code 2.1+):
MCP Tool Search enables lazy loading, reducing context usage by 95% - you can run many MCP servers without hitting context limits.

### Q5: Claude Code keeps forgetting my project context. How do I fix this?

**Answer**:
**Common causes**:
1. **Session restarts**: Each new session starts fresh
2. **Context window limits**: Very large projects exceed memory
3. **Missing project files**: Important context files not read

**Solutions**:

**1. Use project configuration file** (`.claude/project.json`):
```json
{
  "name": "My Project",
  "description": "E-commerce platform with React frontend and Node.js backend",
  "context_files": [
    "README.md",
    "ARCHITECTURE.md",
    "package.json",
    "src/config/app.config.ts"
  ],
  "conventions": {
    "code_style": "Follow Airbnb JavaScript style guide",
    "testing": "Use Jest with React Testing Library",
    "commits": "Use conventional commits format"
  }
}
```

**2. Create CLAUDE.md file** in project root:
```markdown
# Project Context for Claude Code

## Overview
This is an e-commerce platform built with React and Node.js.

## Key Files
- `src/components/`: React UI components
- `src/api/`: Backend API routes
- `src/models/`: Database models

## Coding Standards
- Use TypeScript for all new files
- Write tests for all business logic
- Follow atomic commit practices

## Common Tasks
- Run tests: `npm test`
- Start dev server: `npm run dev`
- Build for production: `npm run build`
```

**3. Use SessionStart hooks** to load context:
```javascript
// .claude/hooks/sessionStart.js
export default function sessionStart() {
  return {
    additionalContext: "Read CLAUDE.md and ARCHITECTURE.md for project overview"
  };
}
```

**4. Explicitly remind Claude**:
- "Remember that this project uses React hooks, not class components"
- "Refer to the coding standards in CONTRIBUTING.md"

**5. Use Memory MCP server**:
Install persistent memory MCP for cross-session context retention.

**Best practice**: Keep project documentation up-to-date so Claude can reference it in each session.

### Q6: How do I use Claude Code for git workflows (commit, PR creation)?

**Answer**:
**Creating commits**:

Simply ask Claude Code to commit your changes:
```
"Review my changes and create a git commit"
```

**What Claude does**:
1. Runs `git status` to see all changes
2. Runs `git diff` to analyze modifications
3. Reviews recent `git log` to match commit message style
4. Analyzes changes to understand purpose
5. Creates descriptive commit message
6. Executes `git add` for relevant files
7. Commits with message including co-author credit

**Example commit message**:
```
Add dark mode toggle to settings page

Implement user preference storage and theme switching
functionality with persistent state management.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

**Creating pull requests**:

Ask Claude to create a PR:
```
"Create a pull request for this feature"
```

**What Claude does**:
1. Checks current branch and tracks remote
2. Runs `git diff main...HEAD` to see all changes since branching
3. Reviews all commits in the PR (not just latest!)
4. Drafts comprehensive PR summary
5. Pushes branch to remote if needed
6. Creates PR using `gh pr create` with summary and test plan

**Example PR**:
```markdown
## Summary
- Add dark mode toggle to settings page
- Implement theme preference persistence
- Update all components for theme support

## Test plan
- [ ] Verify toggle appears in settings
- [ ] Confirm theme persists across sessions
- [ ] Test all pages in both light and dark mode
- [ ] Check mobile responsiveness

🤖 Generated with Claude Code
```

**Best practices**:
- Keep changes focused (easier for Claude to summarize)
- Ensure tests pass before requesting PR
- Review Claude's commit messages before confirming
- Use conventional commit format if your project requires it

**Requirements**:
- `gh` (GitHub CLI) installed for PR creation
- Configured remote repository
- Appropriate repository permissions

### Q7: What's the difference between Claude Code CLI and IDE extensions?

**Answer**:
**Claude Code CLI** (Terminal-based):

**Strengths**:
- Full autonomy: Can execute any terminal command
- Unrestricted access: Complete system and git control
- Flexible workflows: Not limited to IDE capabilities
- Scriptable: Can be automated and integrated into CI/CD

**Limitations**:
- No inline editing visualization (changes shown as diffs)
- Requires terminal comfort
- Less visual feedback

**Best for**:
- Complex multi-step workflows
- Git operations and CI/CD
- System-level tasks
- Developers comfortable with terminal

**IDE Extensions** (VS Code, JetBrains):

**Strengths**:
- Inline editing: See changes directly in editor
- Visual diffs: Side-by-side comparison view
- Integrated experience: No context switching
- Accessible UI: Better for non-terminal users

**Limitations**:
- Constrained to IDE capabilities
- May have restricted file system access
- Less suitable for system-level operations

**Best for**:
- Code editing and refactoring
- Real-time collaboration
- Visual learners
- Developers who prefer GUI

**Can you use both?**
Yes! Many developers use:
- IDE extension for coding and debugging
- CLI for git workflows and system tasks
- Mobile/web for quick tasks on-the-go

**Recommendation**: Start with your preferred environment, then explore others as needs evolve.

### Q8: How do I troubleshoot common Claude Code issues?

**Answer**:
**Issue: "Command not found" after installation**

**Solution**:
```bash
# Reload shell configuration
source ~/.bashrc  # or ~/.zshrc

# Verify PATH includes Claude Code
echo $PATH

# Reinstall if needed
curl -fsSL https://claude.ai/install.sh | bash
```

**Issue: Slow performance or timeouts**

**Solutions**:
1. **Reduce context**: Exclude large files (`.claudeignore`)
2. **Use .claudeignore**: Similar to `.gitignore`
```
node_modules/
*.log
dist/
build/
.git/
```
3. **Enable MCP lazy loading**: Update to Claude Code 2.1+
4. **Close unnecessary MCP servers**: Remove unused servers from `.mcp.json`
5. **Check network**: Slow connections impact API calls

**Issue: Claude makes incorrect changes**

**Solutions**:
1. **Provide more context**: Share relevant files and requirements
2. **Be specific**: "Refactor using React hooks" vs "Make this better"
3. **Review before accepting**: Check diffs carefully
4. **Use iterative refinement**: "That's close, but change X to Y"
5. **Add project conventions**: Use CLAUDE.md to specify standards

**Issue: Authentication failures**

**Solutions**:
```bash
# Log out and back in
claude logout
claude login

# Check account status
claude doctor

# Verify subscription tier
# Ensure Pro/Team/Enterprise/Console account is active
```

**Issue: MCP servers not loading**

**Solutions**:
1. **Verify .mcp.json syntax**: Must be valid JSON
2. **Check environment variables**: Ensure tokens/credentials are set
3. **Test MCP server independently**: Run server command manually
4. **Check logs**: Claude Code logs show MCP loading errors
5. **Update MCP servers**: `npx -y @modelcontextprotocol/server-github@latest`

**Issue: Git operations fail**

**Solutions**:
1. **Verify git installation**: `git --version`
2. **Check repository status**: Ensure you're in a git repository
3. **Configure git user**: `git config user.name` and `user.email` must be set
4. **For PR creation**: Install `gh` CLI and authenticate (`gh auth login`)

**General debugging**:
```bash
# Check Claude Code version and status
claude doctor

# View logs (if available)
claude logs

# Update to latest version
claude update  # Native installer handles this automatically
```

**Get help**:
- Consult [troubleshooting guide](https://smartscope.blog/en/generative-ai/claude/claude-code-troubleshooting-guide/)
- Check [GitHub issues](https://github.com/anthropics/claude-code/issues)
- Visit Anthropic support

### Q9: How can I optimize Claude Code for my specific workflow?

**Answer**:
**1. Create project-specific configuration**:

`.claude/project.json`:
```json
{
  "name": "My App",
  "conventions": {
    "style_guide": "Airbnb JavaScript",
    "testing": "Jest + React Testing Library",
    "commits": "Conventional Commits",
    "branch_naming": "feature/*, bugfix/*, hotfix/*"
  },
  "preferences": {
    "auto_format": true,
    "run_tests_before_commit": true,
    "require_test_coverage": 80
  }
}
```

**2. Use SessionStart hooks for automatic setup**:

```javascript
// .claude/hooks/sessionStart.js
export default function sessionStart({ agent_type }) {
  const skills = [];

  // Activate skills based on project type
  if (isReactProject()) {
    skills.push("react-patterns", "accessibility-checks");
  }

  if (isTestingSession(agent_type)) {
    skills.push("jest-patterns", "test-coverage");
  }

  return {
    activate: skills,
    additionalContext: "Follow conventions in .claude/project.json"
  };
}
```

**3. Create custom slash commands**:

```yaml
# .claude/skills/deploy-staging.yaml
name: Deploy to Staging
command: /deploy-staging
description: Build, test, and deploy to staging environment
steps:
  - Run full test suite
  - If tests pass, build production bundle
  - Deploy to staging server
  - Run smoke tests
  - Send Slack notification
```

**4. Optimize MCP configuration for your tools**:

```json
{
  "mcpServers": {
    "github": { /* ... */ },
    "jira": { /* ... */ },
    "sentry": { /* ... */ },
    "datadog": { /* ... */ }
  }
}
```

**5. Use .claudeignore to focus context**:

```
# Exclude from Claude's context
node_modules/
.git/
dist/
build/
*.log
*.lock
coverage/
.env*
```

**6. Create workflow templates**:

```yaml
# .claude/skills/feature-workflow.yaml
name: Complete Feature Workflow
steps:
  - Create feature branch from main
  - Implement feature based on requirements
  - Write comprehensive tests
  - Run tests and linter
  - Update documentation
  - Create commit with conventional message
  - Push branch and create PR
```

**7. Leverage PreToolUse hooks for consistency**:

```javascript
// .claude/hooks/preToolUse.js
export default function preToolUse({ tool, input }) {
  if (tool === 'Edit' || tool === 'Write') {
    return {
      additionalContext: `
        - Use TypeScript for all new files
        - Include JSDoc comments for functions
        - Follow DRY principles
        - Add unit tests for business logic
      `
    };
  }
}
```

**8. Document common tasks in CLAUDE.md**:

```markdown
## Common Workflows

### Adding a New Feature
1. Create feature branch: `git checkout -b feature/description`
2. Implement in `src/features/`
3. Add tests in `tests/features/`
4. Update API docs in `docs/api/`
5. Run full test suite
6. Create PR with test plan

### Fixing Bugs
1. Create bugfix branch: `git checkout -b bugfix/issue-number`
2. Write failing test that reproduces bug
3. Fix bug
4. Verify test passes
5. Check for related issues
6. Create PR referencing issue
```

**Result**: Claude Code will automatically follow your project conventions, use appropriate tools, and execute workflows consistently.

### Q10: What are the limitations and best practices for Claude Code?

**Answer**:
**Known Limitations**:

1. **Context window**: Even 200K-1M tokens have limits; very large monorepos may exceed capacity
2. **Internet dependency**: Requires constant connection to Anthropic API
3. **Cost**: API usage costs apply (though usually minimal for development)
4. **Code understanding**: May misunderstand complex, undocumented legacy code
5. **Security**: AI sees all code it accesses; avoid sharing secrets/credentials
6. **Autonomous limitations**: Cannot make judgment calls requiring business context
7. **Platform support**: Some features may vary across OS platforms

**Best Practices**:

**Security**:
- Never commit API keys or secrets to code
- Use environment variables for sensitive data
- Review all changes before accepting
- Don't share proprietary code in examples to external forums

**Effectiveness**:
- **Be specific**: "Add error handling for network failures" > "Make this better"
- **Provide context**: Share relevant files, requirements, constraints
- **Iterative refinement**: Start broad, then refine with follow-up requests
- **Review changes**: Always review diffs; AI can make mistakes
- **Use version control**: Commit frequently so you can revert if needed

**Performance**:
- Keep projects focused; exclude unnecessary files
- Use .claudeignore for large dependencies
- Close unused MCP servers
- Break very large tasks into smaller steps

**Collaboration**:
- Document conventions in CLAUDE.md
- Share hooks and skills with team via git
- Use consistent project structure across team
- Include Claude co-author credit in commits for transparency

**Workflow Integration**:
- Use Claude Code for repetitive tasks, not one-off edits
- Combine with manual development (not 100% AI-driven)
- Leverage for boilerplate, tests, docs, refactoring
- Keep creative architecture and design decisions human-driven

**Learning**:
- Ask Claude to explain its changes
- Use as learning tool for new frameworks
- Request best practices and patterns
- Don't blindly accept code without understanding

**When NOT to use Claude Code**:
- One-line changes (faster to edit manually)
- Highly sensitive/classified codebases
- When offline or with unreliable internet
- Creative architectural decisions requiring business judgment
- Final code review (use human reviewers)

**Quote from Anthropic**: "Claude Code is a powerful accelerator, but developers remain in control. It's a tool to enhance productivity, not replace engineering judgment."

---

# Conclusion

## Relationship Between Claude Model and Claude Code

**Claude Model** is the underlying AI intelligence - the large language model that powers various applications including the Claude.ai chat interface, API integrations, and Claude Code.

**Claude Code** is a specialized application built on top of Claude models (primarily Sonnet 4.5 for balance of speed and intelligence) that provides agentic coding capabilities through a CLI interface.

**Analogy**: Claude Model is like an engine; Claude Code is like a car built with that engine specifically for the journey of software development.

## Choosing Between Them

**Use Claude Model API when**:
- Building custom applications
- Integrating AI into your product
- Need flexible, programmatic control
- Require specific model selection (Opus for reasoning, Haiku for speed)

**Use Claude Code when**:
- Developing software and need coding assistance
- Want autonomous task execution
- Need git workflow automation
- Prefer natural language commands over coding API calls

**Use both when**:
- Building AI-powered development tools
- Creating custom coding workflows
- Integrating Claude into your development pipeline

## Getting Started Recommendations

**For Learning Claude Model**:
1. Start with Claude.ai web interface to understand capabilities
2. Read official documentation and prompting guides
3. Try API via Workbench (generates code)
4. Build simple integration with SDK (Python or TypeScript)
5. Experiment with different models for your use case

**For Learning Claude Code**:
1. Install native CLI for your platform
2. Start with simple requests ("Explain this file")
3. Progress to tasks ("Add tests for this component")
4. Set up MCP servers for your tools
5. Create custom hooks/skills for your workflow

**Resources to bookmark**:
- [Claude Platform Docs](https://platform.claude.com/docs/en/home)
- [Claude Code Docs](https://code.claude.com/docs/en/overview)
- [Anthropic News](https://www.anthropic.com/news)
- [GitHub Repository](https://github.com/anthropics/claude-code)

Both technologies are rapidly evolving. Stay updated through official channels, and don't hesitate to experiment - the best way to learn is through hands-on practice!

---

# Sources and References

This comprehensive guide synthesizes information from the following authoritative sources:

## Claude Model Sources

- [Claude (language model) - Wikipedia](https://en.wikipedia.org/wiki/Claude_(language_model))
- [Anthropic - Claude Opus 4.5](https://www.anthropic.com/news/claude-opus-4-5)
- [Claude Platform Documentation](https://platform.claude.com/docs/en/home)
- [Models Overview - Claude Docs](https://platform.claude.com/docs/en/about-claude/models/overview)
- [Prompting Best Practices - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices)
- [Claude API Platform](https://claude.com/platform/api)
- [Anthropic Academy: Build with Claude](https://www.anthropic.com/learn/build-with-claude)
- [Anthropic Claude Timeline](https://www.scriptbyai.com/anthropic-claude-timeline/)
- [Timeline of Anthropic](https://timelines.issarice.com/wiki/Timeline_of_Anthropic)
- [Claude Revenue and Usage Statistics (2026)](https://www.businessofapps.com/data/claude-statistics/)
- [Claude AI Models Compared](https://toolsdock.com/articles/claude-ai-models-comparison/)
- [Claude for Healthcare - Microsoft](https://www.microsoft.com/en-us/industry/blog/healthcare/2026/01/11/bridging-the-gap-between-ai-and-medicine-claude-in-microsoft-foundry-advances-capabilities-for-healthcare-and-life-sciences-customers/)

## Claude Code Sources

- [Claude Code Documentation](https://code.claude.com/docs/en/overview)
- [GitHub - anthropics/claude-code](https://github.com/anthropics/claude-code)
- [Claude Code Product Page](https://claude.com/product/claude-code)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [ClaudeLog - Comprehensive Guide](https://claudelog.com/)
- [The Ultimate Claude Code Resource List (2026)](https://www.scriptbyai.com/claude-code-resource-list/)
- [Claude Code 2.1 NEW Features](https://mlearning.substack.com/p/claude-code-21-new-features-january-2026)
- [Ultimate Guide to Claude MCP Servers](https://generect.com/blog/claude-mcp/)
- [Connect Claude Code to tools via MCP](https://code.claude.com/docs/en/mcp)
- [Claude Code Troubleshooting Guide](https://smartscope.blog/en/generative-ai/claude/claude-code-troubleshooting-guide/)

## Additional Resources

- [Claude for Digital Marketing 2026](https://marketingagent.blog/2026/01/08/how-to-use-claude-for-digital-marketing-in-2026-complete-guide-with-case-studies-strategies/)
- [Claude API Integration Guide 2025](https://collabnix.com/claude-api-integration-guide-2025-complete-developer-tutorial-with-code-examples/)
- [Getting Started with Claude 4 API](https://blog.logrocket.com/getting-started-claude-4-api-developers-walkthrough/)
- [Anthropic Business Breakdown & Founding Story](https://research.contrary.com/company/anthropic)

**Note**: All information is current as of January 2026 based on publicly available sources and official documentation.
