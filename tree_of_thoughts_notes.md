用大语言模型进行深思熟虑的问题解决

## 论文基本信息

**标题**: Tree of Thoughts: Deliberate Problem Solving with Large Language Models

**作者**: Shunyu Yao (Princeton), Dian Yu (Google DeepMind), Jeffrey Zhao (Google DeepMind), Izhak Shafran (Google DeepMind), Thomas L. Griffiths (Princeton), Yuan Cao (Google DeepMind), Karthik Narasimhan (Princeton)

**发表时间**: 2023年5月 (arXiv:2305.10601)

**代码仓库**: https://github.com/princeton-nlp/tree-of-thought-llm

---

## 一、核心问题与动机

### 1.1 现有方法的局限性

当前语言模型在推理时主要采用**从左到右的token级顺序决策**，存在以下问题：

- **缺乏探索能力**：无法探索不同的推理分支
- **缺乏规划能力**：无法进行前瞻性思考
- **无法回溯**：一旦做出决策就无法修改
- **局部决策**：每步决策都是局部的，缺乏全局视角

### 1.2 认知科学启发

论文借鉴了人类认知的**双系统理论**：

- **System 1**：快速、自动、无意识的决策（类似当前LM的token级决策）
- **System 2**：缓慢、深思熟虑、有意识的规划（ToT试图实现的能力）

同时借鉴了**经典AI的问题解决方法**（Newell & Simon, 1950s）：
> "真正的问题解决过程涉及反复使用可用信息来启动探索，探索又反过来揭示更多信息，直到最终发现解决方案。"

---

## 二、Tree of Thoughts (ToT) 框架

### 2.1 核心思想

ToT将问题求解过程建模为**树形搜索**：
- **节点（状态）**: s = [x, z₁...ᵢ]，表示输入和当前的思维序列
- **思维（Thought）**: 连贯的语言序列，作为问题解决的中间步骤
- **搜索**: 通过探索、评估、回溯来找到最优解

### 2.2 与现有方法的对比

| 方法 | 思维生成 | 探索能力 | 评估机制 | 回溯能力 |
|------|---------|---------|---------|---------|
| **IO Prompting** | 直接生成输出 | ❌ | ❌ | ❌ |
| **Chain-of-Thought (CoT)** | 顺序生成推理链 | ❌ | ❌ | ❌ |
| **Self-Consistency (CoT-SC)** | 多次独立采样 | 有限 | 投票 | ❌ |
| **Tree of Thoughts (ToT)** | 结构化分解 | ✅ | ✅ | ✅ |

### 2.3 ToT的四个关键组成部分

#### 2.3.1 思维分解 (Thought Decomposition)

根据问题特性设计合适的中间思维步骤：

- **原则**：
  - 足够"小"：使LM能生成多样化的候选
  - 足够"大"：使LM能评估其对问题解决的贡献

- **示例**：
  - Game of 24：每个思维是一个中间等式（如 "13-9=4"）
  - Creative Writing：思维是写作计划（一段话）
  - Crosswords：思维是要填入的单词

#### 2.3.2 思维生成器 G(pθ, s, k)

两种生成策略：

**策略A：独立采样** (用于思维空间丰富的任务)
```
z⁽ʲ⁾ ~ pθ^CoT(z_{i+1}|s)  (j=1...k)
```
- 使用CoT提示独立生成k个候选思维
- 适用于：Creative Writing（每个思维是段落）

**策略B：序列提议** (用于思维空间受限的任务)
```
[z⁽¹⁾, ..., z⁽ᵏ⁾] ~ pθ^propose(z_{i+1}^{(1...k)}|s)
```
- 使用"提议提示"在同一上下文中生成k个不同思维
- 适用于：Game of 24、Crosswords（每个思维是单词或等式）

#### 2.3.3 状态评估器 V(pθ, S)

使用LM本身作为**启发式函数**来评估状态，这是一种新的方法：
- 传统：编程（如DeepBlue）或学习（如AlphaGo）
- ToT：**基于语言的深思熟虑推理**

两种评估策略：

**策略A：独立评估**
```
V(pθ, S)(s) ~ pθ^value(v|s)  ∀s∈S
```
- 对每个状态独立打分（如1-10分或sure/likely/impossible）
- 评估依据：
  - **前瞻模拟**（如快速确认5+5+14能达到24）
  - **常识推理**（如1,2,3太小无法达到24）

**策略B：交叉投票**
```
V(pθ, S)(s) = 1[s=s*]
s* ~ pθ^vote(s*|S)
```
- 比较多个状态，投票选出最有希望的
- 适用于难以直接评分的情况（如段落连贯性）

#### 2.3.4 搜索算法

**广度优先搜索 (BFS)** - 算法1
```python
for t = 1 to T:
    # 从每个状态生成k个候选思维
    S'_t = {[s,z] | s∈S_{t-1}, z_t∈G(pθ,s,k)}
    # 评估所有候选状态
    V_t = V(pθ, S'_t)
    # 保留前b个最优状态
    S_t = argmax_{S⊂S'_t, |S|=b} Σ_{s∈S} V_t(s)
```
- 适用于：树深度有限（T≤3），可以剪枝到少量候选（b≤5）
- 使用场景：Game of 24, Creative Writing

**深度优先搜索 (DFS)** - 算法2
```python
def DFS(s, t):
    if t > T:
        record output
    for s' in sorted_candidates(G(pθ,s,k)):
        if V(pθ,{s'})(s) > v_threshold:  # 剪枝
            DFS(s', t+1)
        # 否则回溯
```
- 适用于：树深度可变，需要剪枝不可能的分支
- 使用场景：Mini Crosswords

---

## 三、实验设计与结果

### 3.1 Game of 24

**任务描述**：给定4个数字，使用加减乘除运算得到24

**示例**：
- 输入: 4 9 10 13
- 输出: (13-9)*(10-4)=24
- 思维步骤: 
  1. 13-9=4 (剩余: 4,4,10)
  2. 10-4=6 (剩余: 4,6)
  3. 4*6=24

**实验设置**：
- 数据集：从4nums.com抓取，使用901-1000题（相对困难）
- 评估指标：成功率（方程正确且恰好使用每个数字一次）

**结果对比**：

| 方法 | 成功率 |
|-----|--------|
| IO prompt | 7.3% |
| CoT prompt | 4.0% |
| CoT-SC (k=100) | 9.0% |
| **ToT (b=1)** | **45%** |
| **ToT (b=5)** | **74%** |
| IO (best of 100) | 33% |
| CoT (best of 100) | 49% |

**关键发现**：
- CoT甚至比IO表现更差（4.0% vs 7.3%），因为一旦第一步错误就无法挽回
- 约60%的CoT样本在生成第一个等式（前3个token）时就失败
- ToT即使只保留1个候选（b=1）也能达到45%，远超100次CoT采样的最优结果

### 3.2 Creative Writing

**任务描述**：给定4个随机句子，创作一篇4段文章，每段以对应句子结尾

**示例**：
- 输入: 4个随机句子
- 输出: 连贯的4段文章
- 思维: 写作计划（如 "1. Introduce a book that connects..."）

**实验设置**：
- 数据集：从randomwordgenerator.com采样100个输入
- 评估指标：
  - GPT-4打分（1-10分，采样5次取平均）
  - 人工盲测（比较CoT vs ToT）

**结果对比**：

| 方法 | GPT-4平均分 | 人工偏好 |
|-----|------------|---------|
| IO | 6.19 | - |
| CoT | 6.93 | 21/100 |
| **ToT** | **7.56** | **41/100** |
| ToT + Refine | 7.91 | - |

**ToT流程**：
1. 生成5个写作计划，投票选最佳
2. 基于最佳计划生成5篇文章，投票选最佳

**关键发现**：
- 在创意任务中，ToT仍然优于直接生成
- 迭代优化（refine）在自然语言任务中很有效
- 人工评估也确认ToT生成的文章更连贯

### 3.3 Mini Crosswords (5×5)

**任务描述**：根据10个线索（5个横向，5个纵向）填写5×5填字游戏

**实验设置**：
- 数据集：从GooBix抓取156个游戏，测试20个
- 评估指标：
  - 字母正确率（25个/游戏）
  - 单词正确率（10个/游戏）
  - 游戏成功率

**结果对比**：

| 方法 | 成功率 | 字母正确率 | 单词正确率 | 完成游戏数 |
|-----|--------|-----------|-----------|-----------|
| IO | 38.7% | 14% | 0 |
| CoT | 40.6% | 15.6% | 1 |
| **ToT** | **78%** | **60%** | **20/100** |
| ToT + best state | 82.4% | 67.5% | 35/100 |

**ToT-DFS流程**：
1. 将已填单词转换为字母约束
2. 为每个空位提议候选单词并评估置信度
3. 按置信度排序，深度优先探索
4. 如果某个状态下任何剩余线索"不可能"填入，则剪枝回溯
5. 限制搜索步数为100步

**消融实验**：
- **去除剪枝** (-prune): 单词成功率降至41.5%，但能解决3个ToT无法解决的游戏
- **去除回溯** (-backtrack): 单词成功率暴跌至20%

**关键发现**：
- IO和CoT几乎无法解决复杂的填字游戏
- DFS的剪枝启发式虽不完美，但整体效果显著
- 回溯能力对于这类搜索问题至关重要

---

## 四、深度分析

### 4.1 ToT的优势

1. **通用性 (Generality)**
   - IO、CoT、CoT-SC都可以看作ToT的特例（深度和广度受限的树）
   - 可适应各种不同类型的问题

2. **模块化 (Modularity)**
   - 基础LM、思维分解、生成、评估、搜索算法都可以独立变化
   - 易于针对具体问题定制

3. **适应性 (Adaptability)**
   - 可根据问题特性、LM能力、资源约束进行调整
   - 性能-成本可权衡（如调整beam size、投票次数等）

4. **便利性 (Convenience)**
   - 无需额外训练，仅需预训练LM
   - 实施门槛低

### 4.2 不同任务的ToT配置对比

| 维度 | Game of 24 | Creative Writing | Mini Crosswords |
|-----|-----------|-----------------|----------------|
| **思维粒度** | 中间等式 | 写作计划 | 填入单词 |
| **步骤数** | 3 (固定) | 1 | 5-10 (可变) |
| **生成策略** | 序列提议 | 独立采样 | 序列提议 |
| **评估策略** | 独立评估 | 投票 | 独立评估 |
| **搜索算法** | BFS (b=5) | BFS (b=1) | DFS |
| **关键挑战** | 避免早期错误 | 保持连贯性 | 处理约束 |

### 4.3 GPT-4 vs GPT-3.5

为了解ToT在不同LM上的表现，论文测试了GPT-3.5：

**Game of 24**:
| LM | IO | CoT | ToT |
|----|-----|-----|-----|
| GPT-4 | 7.3% | 4.0% | 74% |
| GPT-3.5 | 6% | 3% | 19% |

**Creative Writing**:
| LM | IO | CoT | ToT |
|----|-----|-----|-----|
| GPT-4 | 6.19 | 6.93 | 7.56 |
| GPT-3.5 | 4.47 | 5.16 | 6.62 |

**关键发现**：
- "ToT >>> CoT >>> IO" 在两个LM上都成立
- GPT-3.5 + ToT ≈ GPT-4 + CoT（在Creative Writing上）
- 混合使用：GPT-4生成 + GPT-3.5评估 = 64%（Game of 24）

### 4.4 成本分析

**Game of 24**:
| 方法 | 生成/提示tokens | 每案例成本 | 成功率 |
|-----|----------------|-----------|--------|
| IO (best of 100) | 1.8k / 1.0k | $0.13 | 33% |
| CoT (best of 100) | 6.7k / 2.2k | $0.47 | 49% |
| **ToT** | 5.5k / 1.4k | **$0.74** | **74%** |

**Creative Writing**:
| 方法 | 生成/提示tokens | 每案例成本 |
|-----|----------------|-----------|
| IO | 0.9k / 0.4k | $0.06 |
| CoT | 0.9k / 0.4k | $0.07 |
| **ToT** | 4k / 2.9k | **$0.32** (约5倍) |

**成本优化建议**：
1. 仅在需要深思熟虑推理的困难任务上使用ToT
2. 根据资源约束调整beam size、投票次数等
3. BFS可以在找到解决方案时提前停止
4. 考虑使用更便宜的LM（如GPT-3.5）或混合策略
5. 长期来看，开源LM将使成本大幅降低

---

## 五、相关工作

### 5.1 规划与决策

- **传统方法**：强化学习（需要训练奖励和策略模型）
- **ToT创新**：使用LM自身提供价值估计
- **相关工作**：RAP (Hao et al., 2023) - 类似的MCTS方法，但任务更简单

### 5.2 自我反思

- **Reflexion** (Shinn et al., 2023)：LM为自己的预测提供反馈
- **Self-Debug** (Chen et al., 2023)：基于代码执行结果的反馈
- **Self-Eval Decoding** (Xie et al., 2023)：树搜索 + 自评估，但使用PAL格式（代码表示思维）

**ToT的区别**：
- 更通用的思维表示（不限于代码）
- 可处理创意写作等挑战性任务

### 5.3 程序引导的LM生成

- **LLM+P** (Liu et al., 2023)：将规划委托给经典规划器
- **Newell & Simon (1950s-1970s)**：问题空间搜索理论
- **A\* 搜索**：ToT可看作使用LM自评估作为启发式的A\*算法

---

## 六、局限性与未来方向

### 6.1 局限性

1. **任务适用性**：不是所有任务都需要ToT（GPT-4已很擅长的任务不需要）
2. **计算成本**：比采样方法需要更多资源（5-100倍token生成）
3. **评估不完美**：LM的自评估可能不准确（如填字游戏中无法识别生僻词）
4. **模型依赖**：当前实验主要基于GPT-4，更弱的模型可能表现不佳

### 6.2 未来方向

#### 6.2.1 更先进的搜索算法

- **A\*** 搜索：更智能的启发式函数
- **MCTS** (蒙特卡洛树搜索)：更好地平衡探索与利用
- **束搜索变体**：动态调整beam size

#### 6.2.2 改进评估机制

- **外部知识检索**：解决知识不确定性
- **多模型集成**：结合不同LM的评估
- **学习的启发式**：训练专门的评估模型

#### 6.2.3 微调与训练

- **ToT风格的训练**：
  - 在高层次的反事实决策上微调（而非token级预测）
  - 训练数据：成功/失败的思维轨迹
  - 目标：增强LM的规划和搜索能力

- **思维生成优化**：
  - 生成更多样化的候选思维
  - 提高初始思维的质量

- **评估能力训练**：
  - 更准确地判断状态价值
  - 学习何时剪枝、何时回溯

#### 6.2.4 新应用场景

- **代码生成**：探索不同的实现方案
- **数据分析**：尝试多种分析路径
- **机器人控制**：规划复杂的动作序列
- **科学研究**：探索不同的假设和实验设计

#### 6.2.5 人机协作

- **可解释性**：ToT的推理过程是可读的自然语言
- **人类反馈**：在搜索过程中加入人类指导
- **交互式问题解决**：让用户参与思维生成和评估

#### 6.2.6 效率优化

- **自适应搜索**：根据问题难度动态调整搜索深度
- **增量推理**：复用之前的搜索结果
- **并行化**：同时探索多个分支
- **早停策略**：找到解决方案后立即停止

---

## 七、关键启示

### 7.1 理论启示

1. **双系统思维**：LM需要结合快速联想（System 1）和深思熟虑规划（System 2）
2. **搜索的价值**：即使是强大的LM也能从结构化搜索中受益
3. **语言作为推理工具**：自然语言可以同时用于生成和评估

### 7.2 实践启示

1. **任务分类**：
   - **简单任务**：IO或CoT足够
   - **中等任务**：CoT-SC或ToT (b=1)
   - **困难任务**：ToT完整版（更大的beam size和搜索深度）

2. **实施建议**：
   - 从简单的ToT配置开始（如zero-shot voting）
   - 根据任务特性选择合适的搜索算法
   - 平衡性能和成本
   - 利用领域知识设计思维分解

3. **性能优化**：
   - 精心设计提示词
   - 合理设置超参数（beam size, 投票次数等）
   - 考虑混合使用不同强度的LM
   - 加入领域特定的启发式

### 7.3 研究启示

1. **LM能力边界**：ToT展示了当前LM在复杂推理任务上的潜力和局限
2. **推理范式**：从token级预测到高层次规划是重要的研究方向
3. **评估方法**：LM自评估是一种新的启发式函数来源
4. **经典AI的价值**：结合经典AI方法（如搜索）能显著提升LM能力

---

## 八、实现要点

### 8.1 核心组件实现

#### 思维生成提示示例（Game of 24）

```
Propose possible next steps from the current state:
Input: 4 9 10 13
Current state: 13-9=4 (left: 4 4 10)

Possible next steps:
- 4+4=8 (left: 8 10)
- 4*4=16 (left: 10 16)
- 10-4=6 (left: 4 6)
- 10+4=14 (left: 4 14)
- ...
```

#### 状态评估提示示例（Game of 24）

```
Evaluate if the current state can reach 24:
Current state: 10-4=6 (left: 4 6)

Analysis:
- Can we reach 24 from 4 and 6?
- 4*6=24 ✓
- This state is SURE to reach 24

Rating: sure
```

#### 投票提示示例（Creative Writing）

```
Given several writing plans, vote for the most promising one:

Plan A: Start with a mysterious setting...
Plan B: Introduce the character's background...
Plan C: Begin with an action scene...
...

Analysis:
[Deliberate reasoning about each plan]

Conclusion: The best choice is B
```

### 8.2 搜索算法伪代码

#### BFS实现

```python
def tot_bfs(input, beam_size=5, max_steps=3):
    states = [input]
    
    for step in range(max_steps):
        # 生成候选
        candidates = []
        for state in states:
            thoughts = generate_thoughts(state, k=5)
            for thought in thoughts:
                candidates.append(state + [thought])
        
        # 评估
        values = evaluate_states(candidates)
        
        # 选择top-b
        states = select_top_k(candidates, values, beam_size)
    
    # 返回最佳状态的输出
    best_state = max(states, key=lambda s: values[s])
    return generate_output(best_state)
```

#### DFS实现

```python
def tot_dfs(state, step, max_steps, threshold):
    if step > max_steps:
        return generate_output(state)
    
    # 生成并排序候选
    thoughts = generate_thoughts(state, k=5)
    thoughts = sorted(thoughts, key=lambda t: evaluate(t), reverse=True)
    
    for thought in thoughts:
        new_state = state + [thought]
        
        # 剪枝
        if evaluate(new_state) > threshold:
            result = tot_dfs(new_state, step+1, max_steps, threshold)
            if result:
                return result
    
    # 回溯
    return None
```

### 8.3 提示工程技巧

1. **Few-shot vs Zero-shot**：
   - 思维空间受限时用few-shot（如Game of 24）
   - 开放性任务用zero-shot（如Creative Writing）

2. **提示格式**：
   - 明确输出格式要求
   - 提供清晰的评估标准
   - 包含推理过程示例

3. **温度设置**：
   - 生成思维：较高温度（0.7-0.8）增加多样性
   - 评估/投票：较低温度（0.2-0.5）保证稳定性

---

## 九、总结

### 核心贡献

1. **概念贡献**：提出ToT框架，将经典搜索方法与现代LM结合
2. **方法贡献**：设计了模块化、可扩展的思维生成-评估-搜索流程
3. **实证贡献**：在三个挑战性任务上证明了ToT的有效性
4. **启发贡献**：展示了LM作为通用问题解决器的新可能性

### 关键数字

- Game of 24: 从4% (CoT) 提升到 **74%** (ToT)
- Creative Writing: 从6.93 (CoT) 提升到 **7.56** (ToT)
- Mini Crosswords: 从15.6% (CoT) 提升到 **60%** (ToT)

### 未来展望

ToT代表了将LM从"快速联想"推向"深思熟虑"的重要一步。随着：
- **模型能力**的提升（更强的推理能力）
- **成本效率**的改进（开源模型的发展）
- **方法创新**（更好的搜索算法和评估机制）

我们有望看到ToT在更多实际应用中发挥作用，推动LM向真正的通用问题解决器演进。

---

## 参考资源

- **论文**: https://arxiv.org/html/2305.10601
- **代码**: https://github.com/princeton-nlp/tree-of-thought-llm
- **相关论文**:
  - Chain-of-Thought Prompting (Wei et al., 2022)
  - Self-Consistency (Wang et al., 2022)
  - ReAct (Yao et al., 2022)
  - Reflexion (Shinn et al., 2023)

---

**笔记日期**: 2026年1月12日