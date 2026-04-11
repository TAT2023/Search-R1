# Search-R1：用强化学习训练你的 LLM 进行推理并调用搜索引擎

<div align="center">
  <img src="https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/main/public/logo.png" alt="logo" width="300"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2503.09516">
    <img src="https://img.shields.io/badge/Paper1-blue?style=for-the-badge" alt="Button1"/>
  </a>
  <a href="https://arxiv.org/abs/2505.15117">
    <img src="https://img.shields.io/badge/Paper2-green?style=for-the-badge" alt="Button2"/>
  </a>
  <a href="https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5">
    <img src="https://img.shields.io/badge/Resources-orange?style=for-the-badge" alt="Button3"/>
  </a>
  <a href="https://x.com/BowenJin13/status/1895544294473109889">
    <img src="https://img.shields.io/badge/Tweet-red?style=for-the-badge" alt="Button4"/>
  </a>
  <a href="https://wandb.ai/peterjin/Search-R1-v0.2">
    <img src="https://img.shields.io/badge/Logs-purple?style=for-the-badge" alt="Button5"/>
  </a>
</p>

**Search-R1** 是一个强化学习框架，用于训练**推理与搜索交错进行**的 LLM，也就是能够以协同方式一边推理、一边发起工具调用（例如搜索引擎查询）的语言模型。

Search-R1 基于 [veRL](https://github.com/volcengine/verl) 构建，在 **DeepSeek-R1(-Zero)** 的思路上加入了交错式搜索引擎访问能力，并提供完整开源的 RL 训练流水线。它可以视为 **OpenAI DeepResearch** 的一种开源替代方案，用于推动工具增强型 LLM 推理方向的研究与开发。

我们支持多种 RL 方法（例如 PPO、GRPO、reinforce）、多种 LLM（例如 Llama3、Qwen2.5 等）以及多种搜索引擎（例如本地稀疏/稠密检索器和在线搜索引擎）。

论文：[link1](https://arxiv.org/pdf/2503.09516), [link2](https://arxiv.org/abs/2505.15117)；模型与数据：[link](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5)；Twitter 讨论串：[link](https://x.com/BowenJin13/status/1895544294473109889)；完整实验日志：[prelim](https://wandb.ai/peterjin/Search-R1-open)、[v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[v0.2](https://wandb.ai/peterjin/Search-R1-v0.2)、[v0.3](https://wandb.ai/peterjin/Search-R1-v0.3)。关于这些日志和方法的更多说明见[这里](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/experiment_log.md)。

![single-turn](public/main.png)

## 新闻

- [2025.10] Search-R1 被 Thinking Machines Lab 的首个产品 [Tinker](https://github.com/thinking-machines-lab/tinker-cookbook) 收录。详情见：[Document](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/tool_use/search)。
- [2025.7] Search-R1 已获得 [SkyRL](https://github.com/NovaSky-AI/SkyRL) 支持。详细说明见：[code](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/examples/search)、[Document](https://novasky-ai.notion.site/skyrl-searchr1)。
- [2025.6] Search-R1 已集成到最新版 veRL 中，可以直接利用 veRL 的最新功能。详细说明见：[veRL](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)、[English Document](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md)、[Chinese Document](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md)。
- [2025.5] 第二篇进行详细实证研究的[论文](https://arxiv.org/abs/2505.15117)已发布，并公开日志：[v0.3](https://wandb.ai/peterjin/Search-R1-v0.3)。
- [2025.4] 现已支持 [multinode](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/multinode.md) 训练，可用于 30B+ LLM。
- [2025.4] 现已支持[多种搜索引擎](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md)，包括本地稀疏检索器、带 ANN 索引的本地稠密检索器以及在线搜索引擎。
- [2025.3] 第一篇 Search-R1 [论文](https://arxiv.org/pdf/2503.09516)已发布，并公开日志：[v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[v0.2](https://wandb.ai/peterjin/Search-R1-v0.2)。
- [2025.2] 我们开源了 Search-R1 代码库，并给出了[初步结果](https://wandb.ai/peterjin/Search-R1-open)。

## 导航

- [安装](#installation)
- [快速开始](#quick-start)
- [初步结果](#preliminary-results)
- [推理](#inference)
- [使用你自己的数据集](#use-your-own-dataset)
- [使用你自己的搜索引擎](#use-your-own-search-engine)
- [特性](#features)
- [致谢](#acknowledge)
- [引用](#citations)

<a id="installation"></a>
## 安装

### Search-R1 环境
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# 安装 torch（也可以跳过这一步，让 vllm 自动安装合适的版本）
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# 安装 vllm
pip3 install vllm==0.6.3 # 也可以安装 0.5.4、0.4.2 或 0.3.1

# 安装 verl
pip install -e .

# 安装 flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### 检索器环境（可选）
如果你希望使用本地检索器作为搜索引擎，可以按如下方式安装环境。（建议使用单独的环境。）

```bash
conda create -n retriever python=3.10
conda activate retriever

# 为了安装 faiss-gpu，建议使用 conda 安装 torch
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## 安装 GPU 版 faiss，以保证 RL rollout 的检索效率
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API 依赖
pip install uvicorn fastapi
```

<a id="quick-start"></a>
## 快速开始

下面的示例展示如何在 NQ 数据集上训练一个“推理 + 搜索”LLM，使用 e5 作为检索器、Wikipedia 作为语料库。

(1) 下载索引和语料库。
```bash
save_path=/root/shared-nvme/wiki
nohup python scripts/download.py --save_path $save_path > download.log 2>&1 &
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) 处理 NQ 数据集。
```bash
python scripts/data_process/nq_search.py
```

(3) 启动本地检索服务。
```bash
conda activate retriever
bash retrieval_launch.sh
```

(4) 使用 Llama-3.2-3b-base 运行 RL 训练（PPO）。
```bash
conda activate searchr1
bash train_ppo.sh
```

<a id="preliminary-results"></a>
## 初步结果

(1) 基座模型（llama3.2-3b-base）学会了调用搜索引擎，并获得了更好的性能。

![llama-3b](public/llama32-3b.png)

(2) 基座模型（Qwen2.5-7b-base）可以通过 RL 学会多轮搜索引擎调用与推理。

![multi-turn](public/multi-turn.png)

<a id="inference"></a>
## 推理

#### 你可以用自己感兴趣的问题来体验训练好的 Search-R1 模型。

(1) 启动本地检索服务。
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) 运行推理。
```bash
conda activate searchr1
python infer.py
```

你可以把第 7 行的 `question` 修改成自己感兴趣的问题。

<a id="use-your-own-dataset"></a>
## 使用你自己的数据集

### QA 数据
对于每条问答样本，应构造成如下字典：

```python
data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }
```

具体的数据处理示例可参考 `scripts/data_process/nq_search.py`。

### 语料库

建议将语料库制作成 jsonl 文件，其中每一行都对应一个 passage 字典，包含 `"id"` 和 `"contents"` 两个键。示例可参考 `example/corpus.jsonl`。

`"id"` 对应 passage 的编号，`"contents"` 对应 passage 的内容（`'"' + title + '"\n' + text`）。
例如：

```json
{"id": "0", "contents": "Evan Morris Evan L. Morris (January 26, 1977 \u2013 July 9, 2015) was a lobbyist for Genentech and its parent corporation Roche in Washington."}
...
{"id": "100", "contents": "Three years later, when the United States Exploring Expedition to little-known portions of the globe was organised under Charles Wilkes, Hale was recommended, while yet an undergraduate."}
...
```

**为你的语料库建立索引（可选）。**
如果你希望使用本地检索器作为搜索引擎，可以通过下面的命令为自己的语料库建立索引：

```bash
bash search_r1/search/build_index.sh
```

你可以将 `retriever_name` 和 `retriever_model` 替换为你感兴趣的现成检索器。

<a id="use-your-own-search-engine"></a>
## 使用你自己的搜索引擎

本代码库支持本地稀疏检索器（例如 BM25）、本地稠密检索器（包括使用 GPU 的 flat indexing 和使用 CPU 的 ANN indexing）以及在线搜索引擎（例如 Google、Bing 等）。更多细节见[这里](https://github.com/PeterGriffinJin/Search-R1/tree/main/docs/retriever.md)。

核心思路是将本地或远程搜索引擎服务与主 RL 训练流水线分开部署。

LLM 可以通过调用搜索 API 来使用搜索引擎，例如 `http://127.0.0.1:8000/retrieve`。

你可以参考 `search_r1/search/retriever_server.py`，了解如何启动一个本地检索服务。

<a id="features"></a>
## 特性

- 支持本地稀疏检索器（例如 BM25）。
- 支持本地稠密检索器（包括 flat indexing 和 ANN indexing）。
- 支持 Google Search / Bing Search / Brave Search API 等在线搜索接口。
- 支持现成的神经重排器（neural reranker）。
- 支持多种 RL 方法（例如 PPO、GRPO、reinforce）。
- 支持多种 LLM（例如 Llama3、Qwen2.5 等）。

<a id="acknowledge"></a>
## 致谢

Search-R1 的概念受到 [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1) 和 [TinyZero](https://github.com/Jiayi-Pan/TinyZero/tree/main) 的启发。
其实现基于 [veRL](https://github.com/volcengine/verl) 和 [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main)。
感谢这些团队对开源研究与开发所做的贡献。

## 基于 Search-R1 或受其启发的优秀工作

- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling Deep Research via Reinforcement Learning in Real-world Environments. [![[code]](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)](https://github.com/GAIR-NLP/DeepResearcher)
- [Multimodal-Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1): Incentivizing LMMs to Search. [![[code]](https://img.shields.io/github/stars/EvolvingLMMs-Lab/multimodal-search-r1)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- [OTC](https://arxiv.org/pdf/2504.14870): Optimal Tool Calls via Reinforcement Learning.
- [ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch): Incentivize the Search Capability of LLMs without Searching. [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/ZeroSearch)](https://github.com/Alibaba-NLP/ZeroSearch)
- [IKEA](https://github.com/hzy312/knowledge-r1): Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent. [![[code]](https://img.shields.io/github/stars/hzy312/knowledge-r1)](https://github.com/hzy312/knowledge-r1)
- [Scent of Knowledge](https://arxiv.org/abs/2505.09316): Optimizing Search-Enhanced Reasoning with Information Foraging.
- [AutoRefine](https://www.arxiv.org/pdf/2505.11277): Search and Refine During Think. [![[code]](https://img.shields.io/github/stars/syr-cn/AutoRefine)](https://github.com/syr-cn/AutoRefine)
- [O^2-Searcher](https://arxiv.org/pdf/2505.16582): A Searching-based Agent Model for Open-Domain Open-Ended Question Answering. [![[code]](https://img.shields.io/github/stars/Acade-Mate/O2-Searcher)](https://github.com/Acade-Mate/O2-Searcher)
- [MaskSearch](https://arxiv.org/pdf/2505.20285): A Universal Pre-Training Framework to Enhance Agentic Search Capability. [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/MaskSearch)](https://github.com/Alibaba-NLP/MaskSearch)
- [VRAG-RL](https://arxiv.org/abs/2505.22019): Vision-Perception-Based RAG for Visually Rich Information Understanding. [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/VRAG)](https://github.com/Alibaba-NLP/VRAG)
- [R1-Code-Interpreter](https://arxiv.org/abs/2505.21668): Training LLMs to Reason with Code via SFT and RL. [![[code]](https://img.shields.io/github/stars/yongchao98/R1-Code-Interpreter)](https://github.com/yongchao98/R1-Code-Interpreter)
- [R-Search](https://arxiv.org/abs/2506.04185): Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/QingFei1/R-Search)](https://github.com/QingFei1/R-Search)
- [StepSearch](https://arxiv.org/pdf/2505.15107): Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization. [![[code]](https://img.shields.io/github/stars/Zillwang/StepSearch)](https://github.com/Zillwang/StepSearch)
- [SimpleTIR](https://simpletir.notion.site/report): Stable End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning. [![[code]](https://img.shields.io/github/stars/ltzheng/SimpleTIR)](https://github.com/ltzheng/SimpleTIR)
- [Router-R1](https://arxiv.org/pdf/2506.09033): Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/ulab-uiuc/Router-R1)](https://github.com/ulab-uiuc/Router-R1)
- [SkyRL](https://skyrl.readthedocs.io/en/latest/): A Modular Full-stack RL Library for LLMs. [![[code]](https://img.shields.io/github/stars/NovaSky-AI/SkyRL)](https://github.com/NovaSky-AI/SkyRL)
- [ASearcher](https://arxiv.org/abs/2508.07976): Large-Scale RL for Search Agents. [![[code]](https://img.shields.io/github/stars/inclusionAI/ASearcher)](https://github.com/inclusionAI/ASearcher)
- [ParallelSearch](https://www.arxiv.org/abs/2508.09303): Decompose Query and Search Sub-queries in Parallel with RL. [![[code]](https://img.shields.io/github/stars/Tree-Shu-Zhao/ParallelSearch)](https://github.com/Tree-Shu-Zhao/ParallelSearch)
- [AutoTIR](https://arxiv.org/pdf/2507.21836): Autonomous Tools Integrated Reasoning via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/weiyifan1023/AutoTIR)](https://github.com/weiyifan1023/AutoTIR)
- [verl-tool](https://arxiv.org/pdf/2509.01055): A version of verl to support diverse tool use. [![[code]](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)](https://github.com/TIGER-AI-Lab/verl-tool)
- [Tree-GRPO](https://arxiv.org/abs/2509.21240): Tree Search for LLM Agent Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/AMAP-ML/Tree-GRPO)](https://github.com/AMAP-ML/Tree-GRPO)
- [EviNote-RAG](https://arxiv.org/abs/2509.00877): Enhancing RAG Models via Answer-Supportive Evidence Notes. [![[code]](https://img.shields.io/github/stars/Da1yuqin/EviNoteRAG)](https://github.com/Da1yuqin/EviNoteRAG)
- [GlobalRAG](https://arxiv.org/pdf/2510.20548v1): GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/CarnegieBin/GlobalRAG)](https://github.com/CarnegieBin/GlobalRAG)

<a id="citations"></a>
## 引用

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

```bibtex
@article{jin2025empirical,
  title={An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.15117},
  year={2025}
}
```
