# Research Alpha

> A local research-idea agent that mines the logic of strong AI/ML papers, maps that logic into a user's research domain, and refines ideas through evidence gates, memory, and reviewer-style feedback.

中文说明见下方：[中文 README](#中文-readme)。

## Overview

Research Alpha provides a local CLI and browser GUI for research-idea generation. It is built around a structured evidence workflow:

- collect high-signal papers into a local evidence database;
- extract `Idea Genome Cards` from strong papers;
- aggregate reusable `Pattern Cards`;
- separate Gold evidence, frontier trend context, and user-uploaded domain papers;
- generate candidate ideas with evidence traces and storyline migration;
- refine ideas through persistent multi-turn sessions;
- run a 3-5 round reviewer/refinement loop before finalizing an idea.

## Install

```bash
git clone <your-repo-url>
cd research-alpha

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy the environment template:

```bash
cp .env.example .env
```

Configure one model provider in `.env`:

```bash
RA_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
```

or:

```bash
RA_LLM_PROVIDER=openai
OPENAI_API_KEY=...
```

Optional metadata providers:

```bash
OPENALEX_EMAIL=you@example.com
OPENALEX_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
```

## Initialize

```bash
./ra init
./ra status
./ra ask "Say hello from Research Alpha."
```

## Start The GUI

```bash
./ra gui --host 127.0.0.1 --port 8767
```

Open:

```text
http://127.0.0.1:8767
```

The GUI supports:

- starting a new research-idea search;
- continuing an existing idea session;
- viewing and deleting conversation history;
- checking evidence readiness;
- changing search/model settings;
- opening the evidence library;
- adding papers to the user library by arXiv, DOI, or OpenReview links;
- running manual review or the automatic review loop.

## Build Evidence

For a small local demo with bundled sample papers:

```bash
./ra harvest --file seeds/demo_papers.jsonl
./ra score
./ra genome-build --limit 5 --extractive
./ra pattern-build --limit 5 --extractive
./ra evidence
```

For a stricter LLM-backed evidence base:

```bash
./ra gold-build --remote openalex --venues ICLR,NeurIPS,ICML --from-year 2024 --to-year 2026 --per-venue-year 20
./ra quality-enrich --limit 40 --apply
./ra score
./ra genome-build --limit 8 --provider ds
./ra pattern-build --limit 8 --provider ds
./ra evidence
```

## Generate Ideas

```bash
./ra ideate "research agents for reliable scientific discovery" --remote openalex --limit 20 --ideas 3 --session --lang zh
```

Useful follow-up commands:

```bash
./ra idea-list
./ra idea-rank
./ra idea-session --latest
./ra session-dossier --latest
```

## Refine A Session

List sessions:

```bash
./ra sessions
```

Advance the latest session:

```bash
./ra step --latest "make the novelty claim sharper and the first experiment more falsifiable"
```

Run reviewer feedback:

```bash
./ra reviewer --latest
```

Run the automatic 3-5 round reviewer/refinement loop:

```bash
./ra review-loop --latest --rounds 4 --provider ds
```

View the current session:

```bash
./ra session-view --latest
./ra session-dossier --latest
```

## User Paper Library

The user paper library is for domain knowledge and route learning. It can be managed from the GUI by adding arXiv, DOI, or OpenReview links.

In the system design, user-library papers are kept separate from:

- Gold evidence;
- scoring standards;
- `evidence_basis`;
- `storyline_trace` provenance;
- Pattern/Genome source provenance.

## Common Commands

```bash
./ra gui                         # local web UI
./ra status                      # project/database/provider status
./ra llm                         # inspect or configure provider
./ra ask "test prompt"            # quick provider smoke test
./ra harvest --file ...           # import local paper metadata
./ra gold-build                   # import strict excellent-paper evidence
./ra quality-enrich --apply       # quality metadata annotation
./ra score                        # compute paper weights
./ra top                          # show top papers
./ra trends                       # build frontier trend report
./ra limitations                  # extract recent limitation signals
./ra genome-build                 # build Idea Genome Cards
./ra pattern-build                # build Pattern Cards
./ra evidence                     # readiness audit
./ra audit                        # provenance and grounding audit
./ra cleanup                      # cleanup audit/apply
./ra ideate "..."                 # generate ideas
./ra idea-list                    # list generated ideas
./ra idea-rank                    # rank generated ideas
./ra sessions                     # list idea sessions
./ra step --latest "..."          # refine session
./ra reviewer --latest            # reviewer gate
./ra review-loop --latest         # automatic reviewer/refinement loop
```

Full command list:

```bash
./ra --help
```

## Repository Layout

```text
research_alpha/        core package: CLI pipeline, GUI, DB, evidence logic, LLM adapter
configs/               venue and award-signal configuration
seeds/                 tiny demo paper corpus
tests/                 unit and integration tests
scripts/               live verification helper
docs/                  implementation notes
data/                  local SQLite runtime state, ignored except .gitkeep
outputs/               generated reports/dossiers, ignored except .gitkeep
```

## Test

```bash
python3 -m unittest -b tests.test_app tests.test_llm
```

## Git Hygiene

The repository is configured to ignore local runtime files:

- `.env`
- `.venv/`
- `data/*`
- `outputs/*`
- `work/`
- cache and temporary directories

Keep API keys and generated dossiers out of commits.

---

# 中文 README

> Research Alpha 是一个本地运行的科研 idea agent。它从优秀 AI/ML 论文中抽取逻辑故事线，将其迁移到用户给定的研究领域，并通过证据门、多轮记忆和审稿式反馈来打磨 idea。

## 项目概览

Research Alpha 提供本地 CLI 和浏览器 GUI，围绕一个结构化证据流程工作：

- 收集高信号论文，构建本地 evidence database；
- 从优秀论文中抽取 `Idea Genome Cards`；
- 从多篇 Genome Card 聚合 `Pattern Cards`；
- 区分 Gold evidence、近期趋势论文、用户自建论文库；
- 生成带证据链和故事线迁移轨迹的候选 idea；
- 支持持久化多轮 idea session；
- 在最终 idea 输出前运行 3-5 轮审稿专家反馈和自动改写。

## 安装

```bash
git clone <your-repo-url>
cd research-alpha

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

复制环境变量模板：

```bash
cp .env.example .env
```

在 `.env` 中配置一个模型提供方：

```bash
RA_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
```

或：

```bash
RA_LLM_PROVIDER=openai
OPENAI_API_KEY=...
```

可选论文元数据接口：

```bash
OPENALEX_EMAIL=you@example.com
OPENALEX_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
```

## 初始化

```bash
./ra init
./ra status
./ra ask "Say hello from Research Alpha."
```

## 启动 GUI

```bash
./ra gui --host 127.0.0.1 --port 8767
```

打开：

```text
http://127.0.0.1:8767
```

GUI 支持：

- 新建 research idea search；
- 继续已有 idea session；
- 查看和删除对话记录；
- 查看 evidence readiness；
- 调整检索和模型设置；
- 打开 evidence library；
- 通过 arXiv、DOI、OpenReview 链接添加用户论文；
- 手动审稿或运行自动审稿循环。

## 构建证据库

使用内置 demo 论文数据做本地流程检查：

```bash
./ra harvest --file seeds/demo_papers.jsonl
./ra score
./ra genome-build --limit 5 --extractive
./ra pattern-build --limit 5 --extractive
./ra evidence
```

构建更严格的 LLM 证据库：

```bash
./ra gold-build --remote openalex --venues ICLR,NeurIPS,ICML --from-year 2024 --to-year 2026 --per-venue-year 20
./ra quality-enrich --limit 40 --apply
./ra score
./ra genome-build --limit 8 --provider ds
./ra pattern-build --limit 8 --provider ds
./ra evidence
```

## 生成 idea

```bash
./ra ideate "科研智能体可靠评测" --remote openalex --limit 20 --ideas 3 --session --lang zh
```

常用后续命令：

```bash
./ra idea-list
./ra idea-rank
./ra idea-session --latest
./ra session-dossier --latest
```

## 打磨 session

查看 session：

```bash
./ra sessions
```

推进最新 session：

```bash
./ra step --latest "把 novelty 边界和第一个实验设计说得更能被审稿人验证"
```

运行审稿反馈：

```bash
./ra reviewer --latest
```

运行 3-5 轮自动审稿和改写：

```bash
./ra review-loop --latest --rounds 4 --provider ds
```

查看当前 session：

```bash
./ra session-view --latest
./ra session-dossier --latest
```

## 用户自建论文库

用户论文库用于领域知识和路线学习。可以在 GUI 中通过 arXiv、DOI 或 OpenReview 链接添加论文。

系统会将用户论文库与以下内容分开：

- Gold evidence；
- 评分标准；
- `evidence_basis`；
- `storyline_trace` 来源；
- Pattern/Genome 来源。

## 常用命令

```bash
./ra gui                         # 本地 Web UI
./ra status                      # 项目、数据库、模型配置状态
./ra llm                         # 查看或配置模型
./ra ask "test prompt"            # 模型连通性测试
./ra harvest --file ...           # 导入本地论文元数据
./ra gold-build                   # 严格扩充优秀论文库
./ra quality-enrich --apply       # 质量元数据标注
./ra score                        # 计算论文权重
./ra top                          # 查看高权重论文
./ra trends                       # 生成趋势报告
./ra limitations                  # 抽取近期局限性信号
./ra genome-build                 # 生成 Idea Genome Cards
./ra pattern-build                # 聚合 Pattern Cards
./ra evidence                     # evidence readiness 检查
./ra audit                        # provenance 和 grounding 审计
./ra cleanup                      # 清理审计/执行
./ra ideate "..."                 # 生成 idea
./ra idea-list                    # 查看候选 idea
./ra idea-rank                    # 排序候选 idea
./ra sessions                     # 查看 session
./ra step --latest "..."          # 打磨 session
./ra reviewer --latest            # 审稿 gate
./ra review-loop --latest         # 自动审稿和改写循环
```

完整命令列表：

```bash
./ra --help
```

## 目录结构

```text
research_alpha/        核心代码：CLI、GUI、DB、证据门、LLM adapter
configs/               会议和奖项信号配置
seeds/                 极小 demo 论文数据
tests/                 单元测试和接口测试
scripts/               live verification helper
docs/                  实现笔记
data/                  本地 SQLite 状态，默认不提交
outputs/               生成的报告和 dossier，默认不提交
```

## 测试

```bash
python3 -m unittest -b tests.test_app tests.test_llm
```

