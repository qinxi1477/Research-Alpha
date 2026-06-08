# Research Alpha 简介

Research Alpha 是一个本地运行的科研 idea agent，面向 AI/ML 研究选题与 idea 打磨。它不是简单把论文塞进 prompt，而是先构建优秀论文证据库，从高信号论文中抽取 Idea Genome 和 Pattern，再把这些论文的逻辑故事线迁移到用户给定领域。系统会区分 Gold evidence、近期趋势论文和用户自建论文库，避免把普通趋势论文或用户上传论文误当成评分标准。

核心功能包括：本地 CLI 和 Web GUI、优秀论文库构建、Idea Genome/Pattern 抽取、候选 idea 生成、五维证据驱动评分、多轮 session 记忆、用户论文库、手动审稿 gate，以及 3-5 轮自动审稿专家反馈与改写循环。

GitHub description:

```text
Local research-idea agent that mines top-paper story logic, builds evidence-grounded idea dossiers, and refines them through reviewer-style loops.
```
