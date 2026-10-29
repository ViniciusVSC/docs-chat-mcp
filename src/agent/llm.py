"""Fábrica de modelos LLM (Anthropic Claude ou Ollama local) com detecção de disponibilidade."""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)


def is_ollama_available(base_url: str, timeout: float = 1.5) -> bool:
    """Verifica rapidamente se o servidor Ollama responde na URL configurada."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_chat_model(config: AppConfig | None = None) -> BaseChatModel | None:
    """Retorna o modelo de chat configurado (ChatAnthropic ou ChatOllama).

    Caso nenhuma credencial válida da Anthropic esteja configurada ou o Ollama local
    não esteja acessível, retorna None para que o agente opere com o motor analítico
    local determinístico sem falhar.
    """
    cfg = config or load_config()
    provider = cfg.llm.provider.strip().lower()

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if api_key and not api_key.startswith("sk-ant-api03-sua-chave"):
            return ChatAnthropic(
                model_name=cfg.llm.anthropic_model,
                temperature=cfg.llm.temperature,
                api_key=api_key,  # type: ignore[arg-type]
                timeout=45,
                max_retries=2,
            )
        # Se não houver chave Anthropic, tentar Ollama caso esteja rodando localmente
        if is_ollama_available(cfg.llm.ollama_base_url):
            logger.info("ANTHROPIC_API_KEY ausente; utilizando Ollama local detectado.")
            return ChatOllama(
                model=cfg.llm.ollama_model,
                base_url=cfg.llm.ollama_base_url,
                temperature=cfg.llm.temperature,
            )
        return None

    if provider == "ollama":
        if is_ollama_available(cfg.llm.ollama_base_url):
            return ChatOllama(
                model=cfg.llm.ollama_model,
                base_url=cfg.llm.ollama_base_url,
                temperature=cfg.llm.temperature,
            )
        logger.warning(
            f"Provedor 'ollama' selecionado, mas {cfg.llm.ollama_base_url} não está acessível."
        )
        return None

    return None


def invoke_llm_text(llm: BaseChatModel | None, prompt: str) -> str | None:
    """Invoca o LLM de forma segura e retorna o texto da resposta, ou None em caso de erro/indisponibilidade."""
    if llm is None:
        return None
    try:
        resp: Any = llm.invoke(prompt)
        content = getattr(resp, "content", resp)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(block.get("text", block)) if isinstance(block, dict) else str(block)
                for block in content
            ]
            return "\n".join(parts).strip()
        return str(content).strip()
    except Exception as err:
        logger.warning(f"Falha ao invocar LLM ({ err }); acionando fallback determinístico.")
        return None
