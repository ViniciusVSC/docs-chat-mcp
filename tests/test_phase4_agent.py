"""Testes automatizados para a Fase 4 (Agente LangGraph, roteamento, reescrita de query e citação de fontes)."""

from __future__ import annotations

import pytest

from src.agent.graph import (
    ask_agent_async,
    classify_question,
    classify_question_rule_based,
)


def test_question_routing_classification() -> None:
    """Verifica se o roteador classifica corretamente perguntas de texto, tabela e mistas."""
    assert classify_question_rule_based("Qual o prazo de férias previsto na CLT?") == "text"
    assert classify_question_rule_based("O que a LGPD diz sobre consentimento?") == "text"
    assert classify_question_rule_based("Qual o prazo de arrependimento no CDC?") == "text"
    assert classify_question_rule_based("O que o Marco Civil da Internet estabelece sobre neutralidade da rede?") == "text"

    assert classify_question_rule_based("Qual foi o IPCA acumulado nos últimos 12 meses disponíveis?") == "table"
    assert classify_question_rule_based("Quais os 10 municípios mais populosos do Paraná?") == "table"
    assert classify_question_rule_based("Qual a média do dólar PTAX em 2024?") == "table"
    assert classify_question_rule_based("Quais os 5 municípios com maior PIB em São Paulo?") == "table"

    assert classify_question_rule_based("Qual a relação entre a Selic e a inflação nos últimos 3 anos?") == "mixed"
    assert classify_question("Qual o direito a férias na CLT e qual foi o IPCA acumulado nos últimos 12 meses?", llm=None) == "mixed"


@pytest.mark.asyncio
async def test_agent_end_to_end_text_questions_with_mcp_adapters() -> None:
    """Testa o fluxo de texto (CLT e LGPD) no grafo LangGraph via langchain-mcp-adapters."""
    # 1. Pergunta sobre CLT
    state_clt = await ask_agent_async(
        question="Qual o prazo de férias previsto na CLT?",
        workspace="default",
        llm=None,
    )
    assert state_clt["route"] == "text"
    assert state_clt["docs_relevant"] is True
    assert len(state_clt["retrieved_chunks"]) > 0
    assert "Fontes Citadas" in state_clt["answer"]
    assert "CLT" in state_clt["answer"]
    assert any(s["type"] == "document" and "Art. 130" in s.get("article", "") for s in state_clt["sources"])

    # 2. Pergunta sobre LGPD
    state_lgpd = await ask_agent_async(
        question="O que a LGPD diz sobre consentimento?",
        workspace="default",
        llm=None,
    )
    assert state_lgpd["route"] == "text"
    assert state_lgpd["docs_relevant"] is True
    assert "LGPD" in state_lgpd["answer"] or "13.709" in state_lgpd["answer"]
    assert any(s["type"] == "document" for s in state_lgpd["sources"])


@pytest.mark.asyncio
async def test_agent_end_to_end_table_questions_with_mcp_adapters() -> None:
    """Testa o fluxo de tabelas (IPCA 12 meses e 10 municípios mais populosos do Paraná) via MCP."""
    # 1. IPCA acumulado nos últimos 12 meses
    state_ipca = await ask_agent_async(
        question="Qual foi o IPCA acumulado nos últimos 12 meses disponíveis?",
        workspace="default",
        llm=None,
    )
    assert state_ipca["route"] == "table"
    assert state_ipca["sql_valid"] is True
    assert "bcb_ipca_mensal" in state_ipca["generated_sql"]
    assert state_ipca["sql_results"]["row_count"] == 1
    assert "Fontes Citadas" in state_ipca["answer"]
    assert any(s["type"] == "table" and "bcb_ipca_mensal" in s["table_name"] for s in state_ipca["sources"])

    # 2. 10 municípios mais populosos do Paraná
    state_pr = await ask_agent_async(
        question="Quais os 10 municípios mais populosos do Paraná?",
        workspace="default",
        llm=None,
    )
    assert state_pr["route"] == "table"
    assert state_pr["sql_valid"] is True
    assert state_pr["sql_results"]["row_count"] == 10
    assert "Curitiba" in state_pr["answer"]
    assert "Londrina" in state_pr["answer"]
    assert any(s["type"] == "table" and "ibge_populacao_municipios" in s["table_name"] for s in state_pr["sources"])


@pytest.mark.asyncio
async def test_agent_end_to_end_mixed_question_and_query_rewrite() -> None:
    """Testa o fluxo misto (Selic e inflação nos últimos 3 anos) e o limite de reescrita quando não há base."""
    # 1. Pergunta mista dos critérios de qualidade
    state_mixed = await ask_agent_async(
        question="Qual a relação entre a Selic e a inflação nos últimos 3 anos?",
        workspace="default",
        llm=None,
    )
    assert state_mixed["route"] == "mixed"
    assert state_mixed["sql_valid"] is True
    assert state_mixed["sql_results"]["row_count"] >= 3
    assert "bcb_selic_meta" in state_mixed["generated_sql"]
    assert "bcb_ipca_mensal" in state_mixed["generated_sql"]
    assert "Fontes Citadas" in state_mixed["answer"]
    assert any(s["type"] == "table" for s in state_mixed["sources"])

    # 2. Pergunta fora de escopo em workspace vazio -> deve reescrever até max_rewrites (2) e declarar que não encontrou
    state_empty = await ask_agent_async(
        question="Qual a receita de bolo de cenoura espacial em Marte?",
        workspace="workspace_inexistente_vazio",
        llm=None,
    )
    assert state_empty["route"] == "text"
    assert state_empty["docs_relevant"] is False
    assert state_empty["rewrite_count"] == 2
    assert "Não encontrei informações suficientes" in state_empty["answer"]
