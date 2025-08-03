<img width="2061" height="904" alt="image" src="https://github.com/user-attachments/assets/c213ac32-a9e0-444f-884e-c0eab0930b14" /># Embodied AI

## InternRobotics
https://internrobotics.shlab.org.cn/

<img width="694" height="320" alt="image" src="https://github.com/user-attachments/assets/bc2917b6-8b55-4b05-8f22-997dba20dc6a" />

huggingface: https://huggingface.co/InternRobotics \

github: https://github.com/InternRobotics \


## Lerobot
https://github.com/huggingface/lerobot

<img width="512" height="384" alt="image" src="https://github.com/user-attachments/assets/c7a082bd-708e-401d-8656-a42529ff8865" />



# Agent

## Dobby
https://dobby.now 

[OpenAI ChatGPT 控制 WebCamera](./agent_web_camera.mp4)

## Alita
Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution \

<img width="1938" height="814" alt="image" src="https://github.com/user-attachments/assets/33f6770a-6429-43ad-850c-671216bd7864" />

github: https://github.com/CharlesQ9/Alita


🧐Alita 是一个以“最小预设+最大自我演化”为核心设计哲学的通用智能体系统，摆脱传统预设工具的束缚，依靠动态生成的 MCP（Model Context Protocol）实现任务自适应与能力进化，在 GAIA 基准上超越 OpenAI Deep Research 和 Manus，代表了“极简即未来”的 agent 构建新范式。
➡️中文拓展阅读：https://mp.weixin.qq.com/s/vmp8H-3S_HH6Gvb4dH5FxA
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



智能体经济的终极图景，是一个万亿美元级的自治生态：Agent工厂标准化生产海量垂直领域智能体，Agent市场为其提供能力交易与组合创新的平台；

https://mp.weixin.qq.com/s/wjyElZz-aEMpxGbQ9k3yMw


西门子 Industrial Copilot摘下今年「工业界奥斯卡」赫尔墨斯奖

 https://weibo.com/ttarticle/x/m/show#/id=2309405192298262036504&_wb_client_=1


 阿里第一批企业级 Agent，为什么落在了瓴羊 https://weibo.com/ttarticle/x/m/show#/id=2309405192015255568510&_wb_client_=1


 🔥 Hugging Face 最新开源项目 ScreenEnv，让你轻松打造全栈桌面 AI 智能体！🖥️🤖
还在为部署桌面 GUI 智能体发愁？ScreenEnv 用 Docker 打包了完整的 Ubuntu 桌面环境，不用虚拟机，10 秒内就能搞定一个「能看、能点、能打字、还能录屏」的智能体实验室！💻

支持直接用 Python API 控制，也内置 MCP 协议，AI 系统一键远程接入。搭配 smolagents，只需几行代码，就能训练和部署自己的专属桌面 Agent，轻松实现：
✅ 点击任意坐标
✅ 自动打字、按键
✅ 打开网页或文件
✅ 启动指定应用

支持 GPT-4、Qwen2.5-VL、Claude 等主流大模型，AI 动作全盘掌控🔥！未来还将支持 Android、macOS、Windows，真正实现跨平台 GUI 自动化与评测🎯。

🤗 一行命令体验：pip install screenenv
想了解更多玩法，或者一起参与共建，欢迎加入 Hugging Face 中文社区：Chinese LLMs on Hugging Face


Warp 发布 2.0 Agentic 开发环境#ai创造营# 

第一个面向智能体开发的一站式平台，Terminal-Bench 榜首，SWE-bench 71% 得分

支持多线程：同时让多个智能体并行构建功能、调试和发布代码。

通过文本、文件、图片、URL 等多种方式为智能体提供上下文。

支持语音输入：可用语音下达复杂指令，节省时间。

智能体可自动搜索整个代码库、调用 CLI 工具、查阅 Warp Drive 文档、利用 MCP 服务器获取上下文。 http://t.cn/A6DGrxGY





## Manus





# LLMs

## Qwen

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



## Kimi

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




大语言模型微调终极指南：从基础到突破↓

The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs: An Exhaustive Review of Technologies, Research, Best Practices, Applied Research Challenges and Opportunities

（注：这篇论文知识截止到2024.10）

---

一、微调的背景与意义

大语言模型在预训练阶段积累了丰富的通用语言知识，但要在特定领域（如医疗、金融、工业控制）中发挥作用，仍需通过**微调（Fine-Tuning）**过程来进行能力迁移。微调相比重新训练整个模型，具备以下优势：

- 数据需求低：只需少量任务相关数据；

- 计算成本低：利用已有知识快速适应新任务；

- 任务泛化能力强：能够灵活适应不同领域；

- 部署高效：可用于轻量化推理与边缘部署。

---

二、LLM微调方法分类

1. 无监督微调：无需标签，适用于领域语言适配；

2. 有监督微调（SFT）：使用<输入, 输出>配对数据，常见于问答、摘要任务；

3. 指令微调（Instruction Tuning）：基于prompt引导模型，适用于对话助手类应用；

4. 强化学习微调（如PPO, DPO）：通过用户反馈进行偏好对齐，提升人类喜好度。

---

三、微调全流程七阶段框架

论文提出了一个完整的七阶段微调管线，涵盖从数据到部署的全生命周期：

1. 数据准备：清洗、标注、格式转换、处理不平衡问题；

2. 模型初始化：选择合适的预训练模型与初始参数；

3. 训练环境搭建：配置GPU/TPU环境，定义超参数与优化器；

4. 模型微调：选择全量、LoRA、QLoRA、DoRA、Adapter、HFT等策略；

5. 评估与验证：使用交叉熵、BLEU、ROUGE、准确率等指标评估效果；

6. 模型部署：支持分布式推理、WebGPU、本地化部署、量化优化；

7. 监控与维护：跟踪模型性能，支持知识更新与在线再训练。

---

四、高效微调技术精要

1. 参数高效微调（PEFT）：如LoRA、QLoRA、Adapter，仅训练少量参数即可获得显著效果；

2. 半微调（Half Fine-tuning）：结合全量与冻结层策略，减少过拟合；

3. 专家混合（MoE）与智能体混合（MoA）：通过多子网络组合提升泛化能力；

偏好优化技术：
- PPO：使用强化学习优化人类偏好；
- DPO：直接使用偏好对比数据优化；
- ORPO：以概率比值方式优化偏好选择。

---

五、工业级微调平台

报告还评估了多个主流工业微调平台：

HuggingFace Autotrain、Transformers Trainer

Amazon SageMaker JumpStart & Bedrock

OpenAI Fine-tuning API

NVIDIA NeMo

Optimum、WebLLM等轻量部署工具

---

六、多模态微调与扩展方向

论文最后还涵盖了多模态模型（如VLMs、音频大模型）的微调方法，展示了LLM向图像、音频等领域扩展的能力。例如：
- 医学图文对齐模型
- Whisper音频微调
- CLIP/BLIP等对比学习结构

---

七、挑战与未来方向

报告最后总结了LLM微调面临的关键挑战：

- 可扩展性：数据与算力瓶颈；

- 伦理与安全：偏见传播与数据隐私问题；

- 透明性与可解释性：微调后模型行为不透明；

- 持续更新：支持动态数据与增量学习。

---

总结

这篇报告为LLM微调提供了一套体系化的理论框架与实践指导，是研究者与开发者理解微调复杂性、落地AI产品的宝贵资源。未来，LLM的“低门槛定制能力”将成为推动AI深入垂直行业的核心动力。

访问：arxiv.org/abs/2408.13296



媲美GPT - 4o！Nexus - Gen：融合图像理解、生成与编辑的开源多模态模型！

自 OpenAI GPT - 4o 展现出强大的图片生成能力后，业界对大模型生图能力的探索迅速向全模态方向倾斜，训练全模态模型已然成为研发的重中之重。于是ModelScope 团队重磅推出了可同时完成图像理解、生成和编辑的统一模型 ——Nexus - Gen。

令人惊喜的是，Nexus - Gen 在图像质量和编辑能力上达到了 GPT - 4o 的同等水平，并且团队将成果全方位开源，期望借此引发开发者的广泛讨论，推动 All - to - All 模型领域迈向新高度。

一、技术路线
Nexus - Gen 采用了与 GPT - 4o 类似的 token→(transformer)→(diffusion)→pixels 技术路线。这一设计融合了 SOTA MLLMs 强大的文本预测能力和 Diffusion 模型卓越的图像渲染能力。
作为一个 All - to - All 模型，Nexus - Gen 在输入和输出方面都展现出了卓越的兼容性，同时支持图像和文本模态。自回归 Transformer 输出的文本 Token 经过分类后，能够精准解码成对应的输出文本；而输出的视觉 Token 的 embeddings 则会作为条件，输入到 Vision Decoder 中解码为输出图像。

二、创新策略
魔搭团队创新性地提出了预填充自回归的策略。在训练时，使用可学习特殊 Token 填充对应的图像 Embedding 位置，如此一来，模型便能够学习直接预测任意位置的图像 Token 的能力。
在推理阶段，只要预测到图像的起始 Token BOI，就直接预填充 N 个特殊 Token 到输入序列中。通过这种巧妙的方式，能够确保训练和推理阶段行为的一致性，从而有效消除误差累计。
ModelScope 社区将持续把探索过程中的模型权重、训练数据以及工程框架全部开源，衷心欢迎社区对 Nexus - Gen 和 All - to - All 统一模型的技术未来展开广泛交流。相信在开发者们的共同努力下，Nexus - Gen 将不断进化，为 AI 图像领域带来更多的惊喜与突破。让我们拭目以待，见证 AI 技术在开源力量的推动下，绽放更加绚烂的光彩！
GitHub：http://t.cn/A6gZlR2H


【[355星]Seed1.5-VL：一款强大的视觉-语言基础模型，专为通用多模态理解和推理而设计，能在多种复杂任务中提供卓越表现。亮点：1. 高效架构，仅用5.32亿视觉编码器和200亿参数的MoE LLM，实现顶尖性能；2. 在60个公共基准测试中，38个达到最佳水平；3. 擅长复杂推理、OCR、图解理解、视觉定位、3D空间理解及视频理解等多种能力】  
'Seed1.5-VL, a vision-language foundation model designed to advance general-purpose multimodal understanding and reasoning, achieving state-of-the-art performance on 38 out of 60 public benchmarks.'  
GitHub: github.com/ByteDance-Seed/Seed1.5-VL  

## Google Gemini, Gemmma

Google 发布最新端侧开源模型 Gemma 3n。作为 Gemma 团队最新的小型模型，Gemma 3n 不仅延续了 Gemini Nano 系列高效、轻量的传统，还首次将多模态感知能力融入端侧模型之中，为开发者和最终用户带来了更为丰富和自然的交互体验。

Gemma 3m 在架构设计上体现出了对实际应用场景的深刻理解。团队对模型进行了针对性的定制优化，特别强调了在移动设备等低功耗环境下的高效推理能力。得益于此，Gemma 3n 能够以更低的内存占用（2GB）和更快的响应速度，支持如函数调用、文本与图像的交错处理等复杂任务，并且首次具备了音频和视频输入的理解能力。这种多模态的支持，无疑为端侧 AI 模型的应用边界带来了更大的想象空间，也为开发者在构建更具沉浸感与智能化的应用时提供了坚实基础。

值得关注的是，Gemma 3n 引入了创新的二合一架构，在一个模型内部集成了嵌入式子模型，用户可以根据实际需求动态在高质量与高速度之间切换，无需额外管理多套模型。这一设计理念不仅优化了资源利用，也极大简化了部署和运维流程。

Gemma 3n 的底层技术与 Gemini Nano 一脉相承，意味着开发者可以提前探索和体验下一代 Gemini 模型的架构与能力，为未来的产品升级和技术演进打下基础。在实际应用中，Gemma 3n 已经能够为 Android、Chrome 等主流平台带来更流畅、更智能的端侧 AI 体验。其推理速度、内存占用和多模态输入理解等性能提升，将极大丰富用户在移动设备端的智能交互方式。

此外，Gemma 3n 继续秉承了开放和可访问的理念。Google 通过与社区和合作伙伴的紧密协作，推动了模型的普及和技术生态的繁荣。开发者不仅能够利用主流工具和平台（Google AI Studio、Google GenAI SDK、Higging Face、Kaggle）快速上手 Gemma 3n，也能在此基础上探索更多创新可能。


Qwen3-Coder-flash这个模型，就是Qwen3-Coder-30B-A3B-Instruct确实有点东西啊。

在3060 12G的显卡上，4比特量化，速度可以达到34 token/s，这个速度已经是不错了，反正你不会感觉慢。
如果是4090，那个速度就更快了，不过我没有测。

以前这种模型都是玩具，是用来学习和研究的。
现在真的可以干点事情了。

我让它写了一个小球跳动的，一次性完成，效果不错。
然后又让它写了一个俄罗斯方块，也是一次性完成。
就是左右移动的时候，移动的格子有点大，其余没有问题。

这在前面的版本中，哪怕参数大一些，也是经常搞不定的。

本来还想用gemini cli之类的来试试它的调用工具能力和综合能力的。
结果发现ollama适配Qwen3-Coder-flash的时候，没有适配工具。

这个可以让我们看到两个前景：
1.这个级别的模型，在特定的场景，完全是可以能训练出来的。
目前只是编码场景，其它场景，比如设计、写作、法律、医疗等等也可以做到。

2.可以以极低成本跑在大部分的电脑上，价格完全可以承受。

应该说Qwen3-Coder-flash只是刚刚跨国门槛，未来的潜力还很大。

Qwen团队在训练Qwen3编码模型的时候，利用阿里云的基础设施构建了一个可扩展的系统，能够并行运行 20,000 个独立环境，然后通过获得环境中各种工具的反馈来训练，极大的增强了智能体的能力。
说明这种方法是有效的。

这只是开始，智能体的时代正向我们稳步走来。

模型地址：www.modelscope.cn/models/Qwen/Qwen3-Coder-30B-A3B-Instruct


字节跳动刚刚发布了他们的文本 Diffusion 模型！—— Seed Diffusion Preview！

给不太了解文本 Diffusion 模型的同学，大家都知道现在 transformer 大模型是一个字一个字蹦出来的，而文本Diffusion 模型则是跟图像Diffusion 模型差不多，是一个去噪过程，整段话随机出现文本最后组成所有输出。

Diffusion 文本模型的优点是巨快，字节这个有 每秒 2146 个 token 的速度（应该是现在最快？）。我让它用 Rust 写冒泡排序，几乎是秒出。当然目前 Diffusion 文本模型最大的问题还是性能太低了，很难干活。

目前除了eed Diffusion Preview以外，还有最知名的 Mercury Coder 和 Google 的 Gemini Diffusion.

一会也给大家带来简单的测评。


## Seed
字节的发布blog: seed.bytedance.com/en/seed_diffusion
在线体验地址：studio.seed.ai/exp/seed_diffusion/


继前段时间密集发布了三款 AI 大模型后，Qwen 凌晨又更新了 —— 原本的 Qwen3-30B-A3B 有了一个新版本：Qwen3-30B-A3B-Instruct-2507。

这个新版本是一个非思考模式（non-thinking mode）的新模型。它的亮点在于，仅激活 30 亿（3B）参数，就能展现出与业界顶尖闭源模型，如谷歌的 Gemini 2.5-Flash（非思考模式）和 OpenAI 的 GPT-4o 相媲美的超强实力，这标志着在模型效率和性能优化上的一次重大突破。
继前段时间密集发布了三款 AI 大模型后，Qwen 凌晨又更新了 —— 原本的 Qwen3-30B-A3B 有了一个新版本：Qwen3-30B-A3B-Instruct-2507。

这个新版本是一个非思考模式（non-thinking mode）的新模型。它的亮点在于，仅激活 30 亿（3B）参数，就能展现出与业界顶尖闭源模型，如谷歌的 Gemini 2.5-Flash（非思考模式）和 OpenAI 的 GPT-4o 相媲美的超强实力，这标志着在模型效率和性能优化上的一次重大突破。


## GLM

GPT-5还没影子，但国产这边一个「融合大模型」已经炸出来了！

智谱「悄悄的」发布了最新的旗舰版本模型GLM-4.5，这是一个全新的「融合大模型」，主打Agent Foundation Model。

今天网传的最新消息，GPT-5发布时间又要提前，预计7月底面世！

相较于其他模型竞相「卷参数、刷榜单」，GLM-4.5这次选择了不一样的路线——不跟风，不内卷，而是直接「狙击GPT-5」！

GLM-4.5融合ARC（Agentic/Reasoning/Coding）能力，将推理、编程与Agent能力原生整合，走向更通用、更高效的AI形态。

体验地址：http://t.cn/A6rnBgUU

新一代多模态推理基模Step 3横空出世了！是专为推理时代打造的最适合应用的模型，以最高可达DeepSeek-R1 300%的推理效率击破行业天花板。7月31日，Step 3将正式开源，问鼎开源最强多模推理模型。 

Unsloth AI 新出的，20分钟学会微调大语言模型（LLM）全指南！

你将学会如何：

- 选择合适的模型与训练方法（LoRA、FFT、GRPO）
- 构建数据集与对话模板
- 使用 Unsloth 的 notebook 进行训练
- 在 llama.cpp、Ollama 和 Open WebUI 上运行与部署你的 LLM

文档地址：docs.unsloth.ai

注：视频字幕机翻



#谷歌DeepMind全新MoR架构问世#就在刚刚，KAIST、Mila和谷歌DeepMind团队等放出重磅炸弹——

一个名为Mixture-of-Recursions的全新LLM模型架构。

这个崭新的架构，被业内认为有潜力成为Transformer杀手！

它的推理速度提升2倍，训练FLOP减少，KV缓存内存直接减半。

最终，在135M到1.7B的参数规模下，MoR直接划出了一个新的帕累托前沿：相同的训练FLOPs，但困惑度更低、小样本准确率更高，并且吞吐量提升超过2倍。

全面碾压传统的Transformer！

论文链接：http://t.cn/A6kKLHUm


Chain of Thoughts（CoT）推理革命来了！短而高效的思维链让AI更快更聪明！
- 🧠 CoT热度高涨，但传统推理链往往冗长且易过度思考。
- 🚀 本项目聚焦如何让CoT与大型推理模型（LRMs）更高效，重点技术包括：
  - ✍️ Prompting-based：如“Sketch-of-Thought”“Chain of Draft”等，写得更少，思考更快。
  - 🎯 Budget Control：动态控制推理预算，避免资源浪费。
  - 🌀 Latent Space压缩：用隐空间表达，提升推理密度和速度。
  - 📝 Summarization：压缩推理步骤，保持效果同时缩短链长。
  - ⏭️ Skip Something：跳过不必要步骤，提升效率。
  - 💾 KV Cache管理：优化长序列推理的缓存机制。
  - 🎓 强化学习：用RL控制推理长度，实现智能调节。
- 📅 持续更新最新论文与代码，紧跟2024-2025前沿研究。
- 🏫 背后团队：香港科技大学PEILab，专业且领先。

加速你的AI推理，从这里开始！👉 github.com/Blueyee/Efficient-CoT-LRMs


## Transformer Explainer
Transformer Explainer：为非专业用户打造的交互式Transformer可视化学习工具，基于GPT-2模型，深度剖析生成式文本模型的工作机制。
• 实时本地运行GPT-2，无需安装或特殊硬件，降低学习门槛，立即体验AI预测流程。
• 多层次抽象展示：从数学运算到模型结构，流畅切换，直观理解复杂Transformer机制。
• 支持用户自定义输入，动态观察模型内部各组件和参数如何协同生成下一词。
• 开源免费，促进公众对现代生成式AI技术的普及与教育，助力长期认知积累与方法论深化。
• 一站式工具集成理论与实践，适合教育、研究与自学，推动Transformer透明化与普及化。
深刻洞察Transformer本质，开启生成模型学习新范式。  
了解详情🔗 poloclub.github.io/transformer-explainer/
相关论文🔗 arxiv.fly51fly.workers.dev/abs/2408.04619  

<img width="2061" height="904" alt="image" src="https://github.com/user-attachments/assets/be26bbe4-f1c3-4a6f-b9ba-8fd89288305c" />


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



