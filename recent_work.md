
# VLA




# Agent

https://dobby.now \

[点击观看视频](./agent_web_camera.mp4)

有意思：Alita 本质上代表了一种全新的智能体构建方式：不靠预设，不靠大量人类经验堆叠，而是将自主演化能力交给 agent 自身，最终还benchmark还战胜了manus🧐
《Alita：最小预设与自我演化驱动的通用智能体》

GitHub - CharlesQ9/Alita
🧐Alita 是一个以“最小预设+最大自我演化”为核心设计哲学的通用智能体系统，摆脱传统预设工具的束缚，依靠动态生成的 MCP（Model Context Protocol）实现任务自适应与能力进化，在 GAIA 基准上超越 OpenAI Deep Research 和 Manus，代表了“极简即未来”的 agent 构建新范式。
➡️链接：http://t.cn/A6eSLyOZ
➡️中文拓展阅读：http://t.cn/A6eq05Xi
✨重点

●🧩 摒弃手工预设工具，转向动态MCP生成机制

传统 agent 框架依赖大量预定义工具，而 Alita 仅保留最基础的 web agent，并通过自动生成 MCP（类似工具协议）完成功能扩展，提升泛化能力与组合创造性。
●🧠 核心理念：最小预设（Minimal Predefinition） + 最大自演化（Maximal Self-Evolution）

Alita 模仿人类学习过程，通过任务中自发演化适配外部资源，并保留可复用的“知识块”（MCP Box），构建出高性能、自进化的智能体系统。

●🧠 反传统：不靠工具库的通用智能体范式突破

Alita 拒绝依赖预定义工具与流程，避免了工具覆盖有限、灵活性受限、环境不兼容等三大核心问题，打破目前“堆工具=强 agent”的主流设计惯性。
●🧩 MCP 动态生成机制：让 AI 自己造工具

通过 MCP Brainstorming、ScriptGeneratingTool 和 CodeRunningTool 三个子模块，Alita 可分析任务能力缺口，自行搜索资料、生成代码并封装为 MCP 工具。
●🔁 形成闭环演化系统：从问题 → 工具 → 优化 → 复用

MCP 工具在虚拟环境中验证成功后被记录为可复用资产，失败则触发自动诊断修复，最终构建起能进化与学习的自闭环系统。

●📦 MCP Box = 低成本蒸馏 + Agent 知识共享机制

Alita 可自动总结高效的 MCP 配方并存入 MCP Box，弱小 agent 可借助这些配置显著增强能力，类似“强者教弱者”的知识蒸馏过程，突破传统 LLM 封闭成长路径。
●📈 在 GAIA benchmark 中达成 SOTA 表现

在 GAIA 验证集上 pass@1 达 75.15%，pass@3 高达 87.27%，显著超过 OpenAI Deep Research；即使在强调网页操作能力的测试集上仍保持领先。
●🛠️ 强调 MCP 构建的通用性与抽象性

通过 MCP 抽象（避免过拟合）与防止 MCP 重叠（MCP overload），Alita 能同时实现任务通用性与模块清晰性，在 agent 设计中找到泛化与精度的动态平衡。
●📹 面向复杂多模态任务的“自主创造工具”能力

比如面对视频理解任务，Alita 可自行构造“逐帧处理”工具链，而非依赖开发者预设的 transcript 爬虫，体现出 agent 的超越性创造力。
●🧪 对Humanities等冷门领域也有显著探索

Alita 被用于开发历史问答代理“HistAgent”，在AI for Humanities 中探索通用agent在跨学科推理中的可行性，打破AI仅限于Science类任务的思维局限。
●⚙️ 轻量级工程实现，支持快速复现与开源计划

项目由作者与两位本科生实现，强调“无需复杂架构，也能超越大厂团队”，代码将于近期开放，鼓励开发者构建自己的“Open-Alita”。这对未来的 Auto-Agent 构建、Auto-Tool 生成以及 AGI 进化路径具有启发意义。


# LLMs

https://github.com/QwenLM/Qwen2.5-VL \

https://sebastianraschka.com/blog/2025/the-big-llm-architecture-comparison.html \

<img width="3000" height="2093" alt="image" src="https://github.com/user-attachments/assets/e56a81d3-f818-4afa-ad23-96ebbc22a6a3" />
So, in this article, rather than writing about benchmark performance or training algorithms, I will focus on the architectural developments that define today’s flagship open models.


ThinkDiff: https://mizhenxing.github.io/ThinkDiff/ \
https://github.com/MiZhenxing/ThinkDiff \
<img width="3802" height="918" alt="image" src="https://github.com/user-attachments/assets/8b088401-c3c7-4fb7-ba3d-a7ecb1257228" />


SmolLM3-3B: https://github.com/huggingface/smollm \
Everything about the SmolLM and SmolVLM family of models \
https://huggingface.co/HuggingFaceTB \


Training extremely large neural networks across thousands of GPUs. \
https://www.jeremyjordan.me/distributed-training/


Anthropic's educational courses:\
https://github.com//anthropics/courses

Reinforcement Learning from Human Feedback : \
https://rlhfbook.com \



Kimi

Kimi K2技术报告：最新开源智能体大模型，刷新非思考模式性能新高度。
• 1.04万亿参数MoE架构，激活参数达320亿，基于MuonClip优化器实现15.5万亿高质量token预训练无损失峰值。
• 创新QK-Clip机制，稳定控制注意力logits爆炸，确保大规模训练稳定高效。
• 大规模合成工具调用数据与多阶段强化学习相结合，强化模型自主感知、规划、推理与行动能力。
• 多项权威基准测试领先开源及闭源对手：Tau2-Bench 66.1、ACEBench 76.5、SWE-Bench Verified 65.8，展现卓越编码、数学与推理实力。
• 灵活高效的训练与推理架构支持128k超长上下文，兼顾性能与成本，推动智能体技术前沿。
• 完善安全策略与红队评估，保障生成内容可靠与合规。
• 开源基础及后训练模型检查点，助力社区共建智能体未来。
技术报告👉 github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf
模型下载👉 huggingface.co/moonshotai/Kimi-K2-Instruct




https://arxiv.org/abs/2507.13334 \

论文《A Survey of Context Engineering for Large Language Models》， 上下文工程综述

看完这篇论文，如果你是：
- LLM 应用开发者 —— 你将学会构建更强的“RAG+Memory+Tool”系统
- 研究者 —— 你将了解未来模型能力瓶颈或不在参数，而在上下文处理
- 产品经理 —— 你将理解上下文设计如何成为AI产品竞争力的核心

——
近年来，大语言模型（LLM）如GPT、Claude、Gemini等在各类任务中大放异彩。然而，模型的表现并不总是稳定：同一个任务，不同的提示词（Prompt）可能得出完全不同的结果。这背后的核心问题正是——上下文工程（Context Engineering）。

本文是由中科院计算所、北大、清华、UC Merced等机构联合发布的综述论文，首次系统性地定义并划分了“上下文工程”这一新兴研究领域，并对1400+篇相关论文进行了深入梳理。

一、为什么我们需要“上下文工程”？

传统的Prompt Engineering，主要靠写好一段文字提示来引导模型。这种方法：

- 缺乏系统性

- 不适用于复杂场景（如多轮对话、多工具调用、长文档理解）

- 很难复用与优化

而“上下文工程”则提出：LLM的行为，本质上是由其接收到的信息（Context）决定的。这不仅包括提示词，还包括：

- 检索到的外部知识

- 用户历史对话（记忆）

- 当前任务状态（如工具列表、Agent状态）

- 多模态信息（图像、音频等）

二、上下文工程三大基础模块

1️⃣ 上下文检索与生成（Context Retrieval & Generation）

如何找到任务相关的信息并拼装成上下文？

- Prompt 工程： Few-shot / Zero-shot / Chain-of-Thought（CoT） / Tree-of-Thought（ToT）等推理提示词构造方法。

- 外部检索： 结合 RAG（检索增强生成）从文档、数据库或图谱中动态获取知识。

- 动态组装： 自动挑选、排序、格式化信息，生成符合任务需求的上下文输入。

2️⃣ 上下文处理（Context Processing）

如何让模型更高效地处理超长和结构化信息？

- 长文本处理： 使用Mamba、LongNet、FlashAttention等技术突破O(n²)限制。

- 自我精修： 模型生成后自我评估并反复优化（Self-Refine、Reflexion等）。

- 结构化信息融合： 将表格、图谱、代码等非文本信息嵌入上下文中进行推理。

3️⃣ 上下文管理（Context Management）

如何管理上下文信息，使其可持续、可扩展？

- 记忆系统： 实现长期记忆与短期记忆的分层调度（如MemGPT、MemoryBank）。

- 上下文压缩： 在有限窗口中筛选最关键的信息，如InfiniAttention、StreamingLLM。

- 窗口管理策略： 设计Token淘汰机制（如H2O策略）减少冗余信息。

三、上下文工程的四大系统实践

论文进一步展示了上下文工程如何在大型智能体系统中落地，构建复杂、动态、多模态的AI能力：

✅ RAG系统：结合检索模块进行“知识注入”

如FlashRAG、GraphRAG、ComposeRAG支持结构化知识、多模态输入、多Agent协作等场景。

✅ 记忆系统：支持多轮交互与持续记忆

代表作如MemGPT、MemOS、Self-Controlled Memory等，通过显式记忆模块实现知识持久化。

✅ 工具集成推理：模型调用外部函数、API、环境接口

如Toolformer、ToolLLM、ReAct等支持函数调用链构建与环境交互，实现从问答到执行。

✅ 多智能体系统：多Agent协同完成复杂任务

如CAMEL、AutoGen、CrewAI等构建通信协议、角色扮演、决策协同机制，模拟“团队式思考”。

四、评估方法与未来挑战

论文指出，评估上下文工程系统非常困难：

- 单点评估 vs 系统集成评估

- 缺乏统一基准数据集（如GTA、WebArena、SagaLLM等仍在发展中）

- 面临安全性、健壮性、归因解释等问题

★ 未来研究方向包括：

- 建立理论基础与统一框架

- 多模态与结构化上下文表示方法

- 上下文压缩与调度的自动化优化

- 多Agent协作机制的通用化抽象

五、总结：提示词只是起点，上下文才是核心

这篇论文的重要意义在于，它首次系统性地提出“上下文工程”的概念，把大模型从“提示驱动”推向“系统驱动”的新阶段。

它告诉我们：

> “不是模型不够聪明，而是你没喂它该吃的信息。”

大模型的“聪明”，从来不是靠参数数量决定的，而是靠你给它什么上下文、怎么组织这些上下文。Prompt Engineering 是“术”，Context Engineering 才是“道”。

论文：arxiv.org/abs/2507.13334


# RAG

🔥别再说RAG过时了！是你没抓住核心——多模态RAG才是解锁上下文神器的关键！

✅ 它能同时处理文本、图像、音频、视频，让AI像人类一样跨维度推理，输出更精准、更丰富的答案！

📌 为什么颠覆传统？
🔹 纯文本RAG无法理解现实世界的多模态信息（如：医疗报告+CT扫描图）
🔹 多模态RAG可直接响应复杂请求，例如：“展示新能源汽车充电桩实拍图并解释工作原理”💡

🚀 实战案例演示
👉 分步构建教程：http://t.cn/A6rtuRo6（自动转短链）
👉 5分钟搭建视频：http://t.cn/A6rtu123（自动转短链）

🚀 关键验证：
👉超越文本限制
你完全正确，纯文本RAG限制了上下文。多模态摄取（图像、音频、视频）反映了现实世界的数据复杂性，从而实现了更丰富的语义理解，例如将医学扫描与诊断研究论文进行交叉引用。
👉类人推理
结合模式允许人工智能以人类自然的方式“连接点”（例如，在引用相关新闻的同时描述视频场景）。这弥合了结构化数据和感官上下文之间的差距。
👉动态输出
你给出的例子——比如为可再生能源查询生成信息图表+摘要——展示了当用户收到综合答案而不是零散的链接时，响应效用是如何飙升的。
👉实用可扩展性
您的视频链接展示了快速部署（几分钟解决了采用障碍。Unstructured.io等用于多模式预处理或CLIP/ViT嵌入的工具现在可以在没有大量基础设施的情况下实现这一点。

⚠️ 但非万灵丹:多模态RAG只有在检索骨干网稳健的情况下才能提高准确性。如果对齐不受限制（例如，用于跨模态基础的对比学习），噪声多模态数据比文本更能产生幻觉。

🛠️ 成本/计算权衡:视频/音频索引需要比文本多10-100倍的存储/计算。战略性混合方法（例如，首先提取关键帧/ASR转录本）通常会优化成本。

可采取的后续步骤：
您的资源:
研究了指南（多模态RAG框架）——晚期融合与早期融合架构的出色分解。
现在观看视频教程（YouTube演示）。端到端的LlamaIndex+CLIP实现是🔥 用于快速POC。
实验运行:
#伪码：多模RAG流水线

# Pseudocode: Multimodal RAG pipeline
multimodal_retriever = MultiVectorRetriever(
    text_embedder=OpenAIEmbeddings(),
    image_embedder=CLIPEmbedder(),
    fusion_strategy="weighted_concatenation"  # Your guide's Method 3
)
response = Gemini().generate(
    query="Show urban sustainability projects",
    retrieved_items=multimodal_retriever.fetch("solar_farms.mp4 + policy_docs.pdf")
)
# Outputs: Video clips + PDF summaries + generated analysis

💡社区提问:
您是否使用以下指标对准确性增益进行了基准测试平均倒数排名 对于多模式还是纯文本RAG？我很乐意引用你的结果。

这不仅仅是一次升级，更是一次范式转变。您的工作验证了RAG的下一个前沿是跨模式的上下文编排。赞扬这一贡献🙌. 让我们对这些混合检索策略进行压力测试

💡 马上体验革命性升级，让AI真正听懂你的需求！
→ 关注 @智能时刻 获取每日AI黑科技
→ 加入【智能时刻的铁粉群】http://t.cn/A6rtuRo6 交流实战技巧

#AI创造营 #ai探索计划 #AI学习营 #AI打工人 #热点科普
🔥 评论区喊出你最想破解的多模态难题！👇



# MCP


# Industrial Vision


PyVision: Agentic Vision with Dynamic Tooling \
paper: https://arxiv.org/pdf/2507.07998  \
homepage: https://agent-x.space/pyvision/ \
code: https://github.com/agents-x-project/PyVision \
online demo: https://huggingface.co/spaces/Agents-X/PyVision


# Course

https://ernestryu.com/courses/RL-LLM.html \


CS25: Transformers United V5: https://web.stanford.edu/class/cs25 \

Stanford CS25: V2 I Introduction to Transformers w/ Andrej Karpathy: https://www.youtube.com/watch?v=XfpMkf4rD6E \



# Computer Vision

About
[NeurIPS 2024] Depth Anything V2. A More Capable Foundation Model for Monocular Depth Estimation \
https://github.com/DepthAnything/Depth-Anything-V2



Describe Anything: Detailed Localized Image and Video Captioning \
code: https://github.com/NVlabs/describe-anything \
homepage: https://describe-anything.github.io \


Master AI Agentic Engineering - build autonomous AI Agents: https://github.com/ed-donner/agents \
6 week journey to code and deploy AI Agents with OpenAI Agents SDK, CrewAI, LangGraph, AutoGen and MCP \

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f203ec73-257e-4dd9-b538-5f33c541f175" />



Ovis-U1: Unified Understanding, Generation, and Editing: An unified model that seamlessly integrates multimodal understanding, text-to-image generation, and image editing within a single powerful framework. \
https://github.com/AIDC-AI/Ovis-U1 \
demo : https://huggingface.co/spaces/AIDC-AI/Ovis-U1-3B \


CS336: Language Modeling from Scratch
Stanford / Spring 2025: https://stanford-cs336.github.io/spring2025/

Hunyuan3D-2.1: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 \
3d.hunyuan.tencent.com/ \



