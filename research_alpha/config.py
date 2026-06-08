from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
PROVIDER_ALIASES = {
    "o": "openai",
    "oa": "openai",
    "openai": "openai",
    "d": "deepseek",
    "ds": "deepseek",
    "deepseek": "deepseek",
}


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = ""
    api_key: str = ""
    base_url: str = ""

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider == "deepseek":
            return "deepseek-chat"
        return "gpt-4.1-mini"

    @property
    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.provider == "deepseek":
            return DEFAULT_DEEPSEEK_BASE_URL
        return DEFAULT_OPENAI_BASE_URL


@dataclass
class AppConfig:
    root_dir: Path
    data_dir: Path
    output_dir: Path
    config_dir: Path
    db_path: Path
    llm: LLMConfig
    openalex_api_key: str
    openalex_email: str
    semantic_scholar_api_key: str


LLM_ENV_KEYS = {
    "provider": "RA_LLM_PROVIDER",
    "model": "RA_LLM_MODEL",
    "base_url": "RA_LLM_BASE_URL",
    "generic_api_key": "RA_LLM_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
}


def load_dotenv(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def normalize_provider(value: str) -> str:
    normalized = value.strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized or "openai")


def provider_api_key_env(provider: str) -> str:
    normalized = normalize_provider(provider)
    if normalized == "deepseek":
        return LLM_ENV_KEYS["deepseek_api_key"]
    return LLM_ENV_KEYS["openai_api_key"]


def upsert_dotenv(root: Path, updates: Dict[str, str | None]) -> Path:
    env_path = root / ".env"
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    managed_keys = set(updates.keys())
    retained_lines = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            retained_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key not in managed_keys:
            retained_lines.append(line)

    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
            continue
        normalized = str(value).strip()
        retained_lines.append(f"{key}={normalized}")
        os.environ[key] = normalized

    text = "\n".join(retained_lines).strip()
    env_path.write_text(f"{text}\n" if text else "", encoding="utf-8")
    return env_path


def load_config(root_dir: Path | None = None) -> AppConfig:
    root = Path(root_dir or Path.cwd()).resolve()
    load_dotenv(root)
    data_dir = root / "data"
    output_dir = root / "outputs"
    config_dir = root / "configs"
    db_path = data_dir / "research_alpha.db"

    provider = normalize_provider(os.environ.get("RA_LLM_PROVIDER", "openai"))
    model = os.environ.get("RA_LLM_MODEL", "").strip()
    base_url = os.environ.get("RA_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("RA_LLM_API_KEY", "").strip()
    provider_key = os.environ.get(provider_api_key_env(provider), "").strip()
    if provider_key:
        api_key = provider_key

    llm = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    return AppConfig(
        root_dir=root,
        data_dir=data_dir,
        output_dir=output_dir,
        config_dir=config_dir,
        db_path=db_path,
        llm=llm,
        openalex_api_key=os.environ.get("OPENALEX_API_KEY", "").strip(),
        openalex_email=os.environ.get("OPENALEX_EMAIL", "").strip(),
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip(),
    )
