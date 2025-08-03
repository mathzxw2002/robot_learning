
# VLA




# Agent

https://dobby.now \

./agent_web_camera.mp4

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



