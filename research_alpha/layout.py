from __future__ import annotations

import stat
from pathlib import Path

from research_alpha.config import AppConfig


DEFAULT_ENV_EXAMPLE = """# LLM provider
RA_LLM_PROVIDER=openai
RA_LLM_MODEL=
RA_LLM_BASE_URL=

# Set one or both provider keys
OPENAI_API_KEY=
DEEPSEEK_API_KEY=

# Optional scholarly metadata APIs
OPENALEX_API_KEY=
OPENALEX_EMAIL=
SEMANTIC_SCHOLAR_API_KEY=
"""


DEFAULT_VENUES_YAML = """venues:
  conferences:
    - id: aaai
      display_name: AAAI
      category: ccf-a
      track: ai
    - id: neurips
      display_name: NeurIPS
      category: ccf-a
      track: ml
    - id: acl
      display_name: ACL
      category: ccf-a
      track: nlp
    - id: cvpr
      display_name: CVPR
      category: ccf-a
      track: vision
    - id: iccv
      display_name: ICCV
      category: ccf-a
      track: vision
    - id: icml
      display_name: ICML
      category: ccf-a
      track: ml
    - id: ijcai
      display_name: IJCAI
      category: ccf-a
      track: ai
    - id: iclr
      display_name: ICLR
      category: extended-top-tier
      track: ml
  journals:
    - id: ai
      display_name: Artificial Intelligence
      category: ccf-a
    - id: tpami
      display_name: TPAMI
      category: ccf-a
    - id: ijcv
      display_name: IJCV
      category: ccf-a
    - id: jmlr
      display_name: JMLR
      category: ccf-a
"""


DEFAULT_AWARD_SIGNALS_YAML = """signals:
  best_paper:
    weight: 5.0
  outstanding_paper:
    weight: 4.0
  test_of_time:
    weight: 4.0
  high_citation:
    weight: 3.0
  oral:
    weight: 2.0
  spotlight:
    weight: 1.5
"""


DEFAULT_DEMO_PAPERS_JSONL = """{"title":"Attention Is All You Need","venue":"NeurIPS","year":2017,"award":"oral","citation_count":25000,"influential_citation_count":8000,"abstract":"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder."}
{"title":"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding","venue":"NAACL","year":2019,"award":"high_citation","citation_count":18000,"influential_citation_count":6500,"abstract":"We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers."}
{"title":"Auto-Encoding Variational Bayes","venue":"ICLR","year":2014,"award":"spotlight","citation_count":14000,"influential_citation_count":4200,"abstract":"We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case."}
{"title":"LoRA: Low-Rank Adaptation of Large Language Models","venue":"ICLR","year":2022,"award":"oral","citation_count":5000,"influential_citation_count":1700,"abstract":"We present Low-Rank Adaptation, or LoRA, a parameter-efficient approach to adapting large pre-trained language models."}
{"title":"Learning Transferable Visual Models From Natural Language Supervision","venue":"ICML","year":2021,"award":"best_paper","citation_count":9000,"influential_citation_count":3100,"abstract":"We study the capability of predicting which caption goes with which image on a dataset of 400 million image-text pairs collected from the internet."}
"""


def ensure_layout(config: AppConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.config_dir.mkdir(parents=True, exist_ok=True)


def write_gitkeep(path: Path) -> None:
    marker = path / ".gitkeep"
    if not marker.exists():
        marker.write_text("", encoding="utf-8")


def write_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_local_ra_wrapper(root_dir: Path) -> None:
    wrapper_path = root_dir / "ra"
    if wrapper_path.exists():
        return
    module_search_root = Path(__file__).resolve().parent.parent
    wrapper_text = (
        "#!/usr/bin/env python3\n\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        f"sys.path.insert(0, {str(module_search_root)!r})\n\n"
        "from research_alpha.cli import main\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    raise SystemExit(main())\n"
    )
    wrapper_path.write_text(wrapper_text, encoding="utf-8")
    current_mode = wrapper_path.stat().st_mode
    wrapper_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def scaffold_project_files(config: AppConfig) -> None:
    write_if_missing(config.root_dir / ".env.example", DEFAULT_ENV_EXAMPLE)
    write_if_missing(config.config_dir / "venues.yaml", DEFAULT_VENUES_YAML)
    write_if_missing(config.config_dir / "award_signals.yaml", DEFAULT_AWARD_SIGNALS_YAML)
    write_if_missing(config.root_dir / "seeds" / "demo_papers.jsonl", DEFAULT_DEMO_PAPERS_JSONL)
    write_local_ra_wrapper(config.root_dir)
