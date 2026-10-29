"""Carregamento e validação das configurações do projeto (config.yaml e variáveis de ambiente)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class LLMConfig(BaseModel):
    provider: str = Field(default="anthropic")
    anthropic_model: str = Field(default="claude-3-5-sonnet-latest")
    ollama_model: str = Field(default="llama3.1:8b")
    ollama_base_url: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.0)


class EmbeddingsConfig(BaseModel):
    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


class StorageConfig(BaseModel):
    data_dir: str = Field(default="data")
    docs_dir: str = Field(default="data/docs")
    tables_dir: str = Field(default="data/tables")
    chroma_dir: str = Field(default="data/chroma_db")
    default_workspace: str = Field(default="default")


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)


class DuckDBConfig(BaseModel):
    max_rows: int = Field(default=500)
    timeout_seconds: int = Field(default=10)


class AgentConfig(BaseModel):
    max_rewrite_retries: int = Field(default=2)
    top_k: int = Field(default=5)


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    duckdb: DuckDBConfig = Field(default_factory=DuckDBConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Carrega o arquivo config.yaml e aplica overrides de variáveis de ambiente."""
    if config_path is None:
        env_cfg = os.getenv("CONFIG_PATH")
        if env_cfg:
            path = Path(env_cfg)
        else:
            path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        path = Path(config_path)

    raw_data: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw_data = loaded

    cfg = AppConfig(**raw_data)

    if os.getenv("OLLAMA_BASE_URL"):
        cfg.llm.ollama_base_url = os.environ["OLLAMA_BASE_URL"]

    if os.getenv("DATA_DIR"):
        base = Path(os.environ["DATA_DIR"])
        cfg.storage.data_dir = str(base)
        cfg.storage.docs_dir = str(base / "docs")
        cfg.storage.tables_dir = str(base / "tables")
        cfg.storage.chroma_dir = str(base / "chroma_db")

    return cfg
