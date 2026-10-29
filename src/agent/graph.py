"""Grafo LangGraph do assistente docs-chat-mcp consumindo o servidor MCP via langchain-mcp-adapters."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from src.agent.llm import get_chat_model, invoke_llm_text
from src.config import AppConfig, load_config
from src.mcp_server.server import mcp as fastmcp_server
from src.mcp_server.sql_engine import UnsafeSQLError, validate_readonly_sql

logger = logging.getLogger(__name__)

RouteType = Literal["text", "table", "mixed"]


class AgentState(TypedDict, total=False):
    """Estado tipado do grafo LangGraph."""

    question: str
    workspace: str
    route: RouteType

    # Fluxo de Texto (Documentos)
    search_query: str
    doc_filter: str | None
    retrieved_chunks: list[dict[str, Any]]
    docs_relevant: bool
    rewrite_count: int
    max_rewrites: int

    # Fluxo de Tabelas (SQL / DuckDB)
    tables_schema: list[dict[str, Any]]
    generated_sql: str
    sql_valid: bool
    sql_error: str | None
    sql_results: dict[str, Any]
    additional_sql_runs: list[dict[str, Any]]

    # Resposta Final e Fontes
    answer: str
    sources: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Heurísticas e Prompts de Apoio (Classificação, Filtro de Lei, Reescrita e SQL)
# ---------------------------------------------------------------------------

UF_NAME_TO_SIGLA: dict[str, str] = {
    "acre": "AC",
    "alagoas": "AL",
    "amapa": "AP",
    "amapá": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceara": "CE",
    "ceará": "CE",
    "distrito federal": "DF",
    "espirito santo": "ES",
    "espírito santo": "ES",
    "goias": "GO",
    "goiás": "GO",
    "maranhao": "MA",
    "maranhão": "MA",
    "mato grosso do sul": "MS",
    "mato grosso": "MT",
    "minas gerais": "MG",
    "paraiba": "PB",
    "paraíba": "PB",
    "parana": "PR",
    "paraná": "PR",
    "pernambuco": "PE",
    "piaui": "PI",
    "piauí": "PI",
    "rio de janeiro": "RJ",
    "rio grande do norte": "RN",
    "rio grande do sul": "RS",
    "rondonia": "RO",
    "rondônia": "RO",
    "roraima": "RR",
    "santa catarina": "SC",
    "sao paulo": "SP",
    "são paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
    "para": "PA",
    "pará": "PA",
}


def detect_doc_filter_from_question(question: str) -> str | None:
    """Identifica se a pergunta menciona especificamente uma das leis conhecidas."""
    q = question.lower()
    if any(k in q for k in ("clt", "consolidação das leis do trabalho", "trabalhador", "empregado", "férias", "fgts", "aviso prévio", "justa causa", "jornada de trabalho", "carteira de trabalho")):
        if not any(k in q for k in ("cdc", "lgpd", "marco civil")):
            return "clt"
    if any(k in q for k in ("lgpd", "proteção de dados", "dados pessoais", "dados sensíveis", "anpd", "controlador", "encarregado")):
        if not any(k in q for k in ("clt", "cdc", "marco civil")):
            return "lgpd"
    if any(k in q for k in ("cdc", "defesa do consumidor", "código do consumidor", "fornecedor", "vício do produto", "fato do produto", "arrependimento")):
        if not any(k in q for k in ("clt", "lgpd", "marco civil")):
            return "cdc"
    if any(k in q for k in ("marco civil", "12.965", "neutralidade da rede", "registros de conexão", "guarda de registros", "provedor de conexão")):
        if not any(k in q for k in ("clt", "cdc", "lgpd")):
            return "marco_civil_internet"
    return None


def classify_question_rule_based(question: str) -> RouteType:
    """Classifica deterministicamente a pergunta entre 'text', 'table' ou 'mixed'."""
    q = question.lower()

    text_keywords = [
        "clt",
        "cdc",
        "lgpd",
        "marco civil",
        "lei",
        "artigo",
        "art.",
        "decreto",
        "férias",
        "consentimento",
        "consumidor",
        "fornecedor",
        "dados pessoais",
        "dados sensíveis",
        "neutralidade da rede",
        "justa causa",
        "aviso prévio",
        "arrependimento",
        "garantia legal",
        "jornada",
        "demissão",
        "provedor",
        "titular",
        "cláusula",
        "contrato de trabalho",
    ]
    table_keywords = [
        "ipca",
        "selic",
        "inflação",
        "inflacao",
        "dólar",
        "dolar",
        "ptax",
        "câmbio",
        "cambio",
        "pib",
        "população",
        "populacao",
        "populosos",
        "populoso",
        "habitantes",
        "município",
        "municipio",
        "municípios",
        "municipios",
        "ibge",
        "banco central",
        "bcb",
        "taxa de juros",
        "acumulado",
        "média",
        "media",
        "soma",
        "ranking",
        "maiores",
        "menores",
        "tabela",
        "csv",
    ]

    has_text = any(k in q for k in text_keywords)
    has_table = any(k in q for k in table_keywords)

    # Perguntas sobre relação entre Selic e inflação/IPCA ou que misturam lei + indicadores
    if "relação" in q and ("selic" in q or "inflação" in q or "ipca" in q):
        return "mixed"
    if has_text and has_table:
        return "mixed"
    if has_table:
        return "table"
    return "text"


def classify_question(question: str, llm: BaseChatModel | None = None) -> RouteType:
    """Classifica a pergunta usando o LLM (se disponível) combinado com regras de segurança."""
    rule_route = classify_question_rule_based(question)
    if llm is None:
        return rule_route

    prompt = (
        "Classifique a pergunta do usuário em exatamente UMA das três categorias:\n"
        "- 'text': perguntas sobre documentos, leis (CLT, CDC, LGPD, Marco Civil da Internet), direitos, prazos legais ou conceitos.\n"
        "- 'table': perguntas numéricas/estatísticas sobre tabelas (Selic, IPCA, Dólar PTAX, População de municípios do IBGE, PIB municipal, UFs).\n"
        "- 'mixed': perguntas que combinam documentos/conceitos e tabelas OU que pedem análise conjunta e conceitual de múltiplos indicadores (ex: relação entre Selic e inflação, ou cruzamento de leis com dados econômicos).\n\n"
        f"Pergunta: {question}\n"
        "Responda APENAS com uma palavra: text, table ou mixed."
    )
    resp = invoke_llm_text(llm, prompt)
    if resp:
        clean = resp.strip().lower()
        # Se a regra determinística detectou claramente mixed, preservamos mixed
        if rule_route == "mixed":
            return "mixed"
        for candidate in ("mixed", "table", "text"):
            if re.search(rf"\b{candidate}\b", clean):
                return candidate  # type: ignore[return-value]
    return rule_route


def expand_rewritten_query_deterministic(question: str, attempt: int) -> str:
    """Gera uma query reescrita enriquecida com termos técnicos/jurídicos para melhorar o recall."""
    q_lower = question.lower()
    expansions: list[str] = [question]

    if "férias" in q_lower or "ferias" in q_lower:
        expansions.append("Art. 129 Art. 130 Art. 134 CLT período aquisitivo concessão 12 meses 30 dias corridos faltas")
    if "consentimento" in q_lower:
        expansions.append("Art. 5 XII Art. 7 I Art. 8 LGPD manifestação livre informada inequívoca titular dados pessoais revogação")
    if "arrependimento" in q_lower or "desistir" in q_lower or "compras fora" in q_lower:
        expansions.append("Art. 49 CDC consumidor desistir contrato prazo de 7 dias a contar de sua assinatura ou do ato de recebimento")
    if "vício" in q_lower or "defeito" in q_lower or "garantia" in q_lower:
        expansions.append("Art. 18 Art. 26 CDC prazo reclamar vícios aparentes 30 dias 90 dias produtos duráveis")
    if "justa causa" in q_lower:
        expansions.append("Art. 482 CLT constituem justa causa para rescisão do contrato de trabalho pelo empregador")
    if "aviso prévio" in q_lower or "aviso previo" in q_lower:
        expansions.append("Art. 487 CLT aviso prévio trinta dias")
    if "jornada" in q_lower or "horas extras" in q_lower or "duração do trabalho" in q_lower:
        expansions.append("Art. 58 Art. 59 CLT duração normal do trabalho oito horas diárias")
    if "dados sensíveis" in q_lower or "dado sensível" in q_lower:
        expansions.append("Art. 5 II Art. 11 LGPD dado pessoal sobre origem racial ou étnica convicção religiosa saúde biométrico")
    if "neutralidade" in q_lower:
        expansions.append("Art. 9 Marco Civil da Internet tratamento isonômico pacotes de dados")
    if "guarda" in q_lower or "registros de conexão" in q_lower or "acesso a aplicações" in q_lower:
        expansions.append("Art. 13 Art. 15 Marco Civil da Internet guarda de registros de conexão 1 ano 6 meses")
    if "selic" in q_lower or "inflação" in q_lower or "ipca" in q_lower:
        expansions.append("taxa de juros básica política monetária controle da inflação crédito consumo")

    if attempt >= 2:
        expansions.append("disposições gerais direitos deveres prazos artigos")

    return " ".join(expansions)


def build_sql_for_question(
    question: str,
    tables_schema: list[dict[str, Any]],
    llm: BaseChatModel | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Gera a consulta SQL principal (e consultas auxiliares se a pergunta comparar múltiplas séries)."""
    q = question.lower()
    extra_queries: list[tuple[str, str]] = []

    # Caso 1: Relação entre Selic e Inflação (IPCA) nos últimos N anos
    if ("selic" in q and ("inflação" in q or "inflacao" in q or "ipca" in q)):
        years_match = re.search(r"(\d+)\s*anos?", q)
        n_years = int(years_match.group(1)) if years_match else 3
        months = n_years * 12
        sql_main = f"""
WITH ipca_recente AS (
    SELECT
        SUBSTR(CAST(data AS VARCHAR), 1, 4) AS ano,
        valor AS ipca_mensal
    FROM bcb_ipca_mensal
    ORDER BY data DESC
    LIMIT {months}
),
ipca_anual AS (
    SELECT
        ano,
        COUNT(*) AS meses_ipca,
        ROUND((EXP(SUM(LN(1.0 + ipca_mensal / 100.0))) - 1.0) * 100.0, 2) AS ipca_acumulado_pct
    FROM ipca_recente
    GROUP BY ano
),
selic_anual AS (
    SELECT
        SUBSTR(CAST(data AS VARCHAR), 1, 4) AS ano,
        ROUND(AVG(valor), 2) AS selic_media_aa,
        ROUND(MIN(valor), 2) AS selic_min_aa,
        ROUND(MAX(valor), 2) AS selic_max_aa
    FROM bcb_selic_meta
    GROUP BY SUBSTR(CAST(data AS VARCHAR), 1, 4)
)
SELECT
    i.ano,
    i.meses_ipca,
    i.ipca_acumulado_pct,
    s.selic_media_aa,
    s.selic_min_aa,
    s.selic_max_aa
FROM ipca_anual i
LEFT JOIN selic_anual s ON i.ano = s.ano
ORDER BY i.ano DESC
""".strip()
        return sql_main, extra_queries

    # Caso 2: IPCA acumulado nos últimos N meses (padrão 12) ou em um ano específico
    if "ipca" in q or "inflação" in q or "inflacao" in q:
        year_match = re.search(r"\b(20\d{2})\b", q)
        months_match = re.search(r"(\d+)\s*meses", q)
        if year_match and "últimos" not in q and "ultimos" not in q:
            yr = year_match.group(1)
            sql = f"""
SELECT
    '{yr}' AS ano,
    MIN(CAST(data AS VARCHAR)) AS data_inicial,
    MAX(CAST(data AS VARCHAR)) AS data_final,
    COUNT(*) AS meses,
    ROUND((EXP(SUM(LN(1.0 + valor / 100.0))) - 1.0) * 100.0, 2) AS ipca_acumulado_composto_pct,
    ROUND(SUM(valor), 2) AS ipca_soma_simples_pct,
    ROUND(MAX(valor), 2) AS maior_variacao_mensal_pct,
    ROUND(MIN(valor), 2) AS menor_variacao_mensal_pct
FROM bcb_ipca_mensal
WHERE SUBSTR(CAST(data AS VARCHAR), 1, 4) = '{yr}'
""".strip()
            return sql, extra_queries
        else:
            n_months = int(months_match.group(1)) if months_match else 12
            sql = f"""
WITH ultimos_meses AS (
    SELECT data, valor
    FROM bcb_ipca_mensal
    ORDER BY data DESC
    LIMIT {n_months}
)
SELECT
    MIN(CAST(data AS VARCHAR)) AS data_inicial,
    MAX(CAST(data AS VARCHAR)) AS data_final,
    COUNT(*) AS meses_considerados,
    ROUND((EXP(SUM(LN(1.0 + valor / 100.0))) - 1.0) * 100.0, 2) AS ipca_acumulado_composto_pct,
    ROUND(SUM(valor), 2) AS ipca_soma_simples_pct,
    ROUND(AVG(valor), 2) AS ipca_media_mensal_pct
FROM ultimos_meses
""".strip()
            return sql, extra_queries

    # Caso 3: Taxa Selic
    if "selic" in q:
        year_match = re.search(r"\b(20\d{2})\b", q)
        if year_match:
            yr = year_match.group(1)
            sql = f"""
SELECT
    '{yr}' AS ano,
    MIN(CAST(data AS VARCHAR)) AS data_inicial,
    MAX(CAST(data AS VARCHAR)) AS data_final,
    ROUND(AVG(valor), 2) AS selic_media_aa,
    ROUND(MIN(valor), 2) AS selic_minima_aa,
    ROUND(MAX(valor), 2) AS selic_maxima_aa
FROM bcb_selic_meta
WHERE SUBSTR(CAST(data AS VARCHAR), 1, 4) = '{yr}'
""".strip()
        else:
            sql = """
SELECT
    CAST(data AS VARCHAR) AS data,
    valor AS selic_meta_aa
FROM bcb_selic_meta
ORDER BY data DESC
LIMIT 10
""".strip()
        return sql, extra_queries

    # Caso 4: Dólar PTAX
    if "dólar" in q or "dolar" in q or "ptax" in q or "câmbio" in q or "cambio" in q:
        year_match = re.search(r"\b(20\d{2})\b", q)
        if year_match:
            yr = year_match.group(1)
            sql = f"""
SELECT
    '{yr}' AS ano,
    MIN(CAST(data AS VARCHAR)) AS data_inicial,
    MAX(CAST(data AS VARCHAR)) AS data_final,
    ROUND(AVG(valor), 4) AS dolar_ptax_media,
    ROUND(MIN(valor), 4) AS dolar_ptax_minimo,
    ROUND(MAX(valor), 4) AS dolar_ptax_maximo
FROM bcb_dolar_ptax_venda
WHERE SUBSTR(CAST(data AS VARCHAR), 1, 4) = '{yr}'
""".strip()
        else:
            sql = """
SELECT
    CAST(data AS VARCHAR) AS data,
    ROUND(valor, 4) AS dolar_ptax_venda
FROM bcb_dolar_ptax_venda
ORDER BY data DESC
LIMIT 10
""".strip()
        return sql, extra_queries

    # Detectar filtro de UF na pergunta (por nome do estado ou sigla)
    detected_uf: str | None = None
    for state_name, sigla in sorted(UF_NAME_TO_SIGLA.items(), key=lambda x: -len(x[0])):
        if state_name in q:
            detected_uf = sigla
            break
    if not detected_uf:
        sigla_match = re.search(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b", question)
        if sigla_match:
            detected_uf = sigla_match.group(1).upper()

    top_n_match = re.search(r"\b(\d+)\s+(?:munic[íi]pios|cidades|maiores|primeiros)", q)
    limit_n = int(top_n_match.group(1)) if top_n_match else 10

    # Caso 5: PIB per capita (cruzamento de PIB e População)
    if "pib" in q and ("per capita" in q or "por habitante" in q):
        where_uf = f"WHERE p.uf_sigla = '{detected_uf}'" if detected_uf else ""
        sql = f"""
SELECT
    p.municipio_nome,
    p.uf_sigla,
    p.pib_mil_reais,
    pop.populacao_estimada,
    ROUND((p.pib_mil_reais * 1000.0) / NULLIF(pop.populacao_estimada, 0), 2) AS pib_per_capita_reais
FROM ibge_pib_municipios p
JOIN ibge_populacao_municipios pop ON p.municipio_id = pop.municipio_id
{where_uf}
ORDER BY pib_per_capita_reais DESC
LIMIT {limit_n}
""".strip()
        return sql, extra_queries

    # Caso 6: PIB dos municípios
    if "pib" in q or "produto interno bruto" in q:
        if "estado" in q or "uf" in q or ("soma" in q and not detected_uf):
            sql = f"""
SELECT
    uf_sigla,
    uf_nome,
    ano,
    COUNT(*) AS total_municipios,
    ROUND(SUM(pib_mil_reais) / 1000000.0, 2) AS pib_total_bilhoes_reais
FROM ibge_pib_municipios
GROUP BY uf_sigla, uf_nome, ano
ORDER BY pib_total_bilhoes_reais DESC
LIMIT {limit_n}
""".strip()
            return sql, extra_queries

        where_uf = f"WHERE uf_sigla = '{detected_uf}'" if detected_uf else ""
        sql = f"""
SELECT
    municipio_nome,
    uf_sigla,
    ano,
    pib_mil_reais,
    ROUND(pib_mil_reais / 1000000.0, 2) AS pib_bilhoes_reais
FROM ibge_pib_municipios
{where_uf}
ORDER BY pib_mil_reais DESC
LIMIT {limit_n}
""".strip()
        return sql, extra_queries

    # Caso 7: Quantidade de municípios por UF ou região
    if ("quantos municípios" in q or "quantos municipios" in q or "total de municípios" in q):
        if detected_uf:
            sql = f"""
SELECT
    uf_sigla,
    uf_nome,
    COUNT(*) AS total_municipios
FROM ibge_municipios
WHERE uf_sigla = '{detected_uf}'
GROUP BY uf_sigla, uf_nome
""".strip()
        else:
            sql = """
SELECT
    uf_sigla,
    uf_nome,
    regiao_nome,
    COUNT(*) AS total_municipios
FROM ibge_municipios
GROUP BY uf_sigla, uf_nome, regiao_nome
ORDER BY total_municipios DESC
LIMIT 27
""".strip()
        return sql, extra_queries

    # Caso 8: População de municípios / estado
    if any(k in q for k in ("população", "populacao", "populosos", "populoso", "habitantes")):
        if ("população total" in q or "populacao total" in q or "soma da população" in q) and detected_uf:
            sql = f"""
SELECT
    uf_sigla,
    uf_nome,
    ano,
    COUNT(*) AS total_municipios,
    SUM(populacao_estimada) AS populacao_total_uf
FROM ibge_populacao_municipios
WHERE uf_sigla = '{detected_uf}'
GROUP BY uf_sigla, uf_nome, ano
""".strip()
            return sql, extra_queries

        where_uf = f"WHERE uf_sigla = '{detected_uf}'" if detected_uf else ""
        order_dir = "ASC" if ("menos populosos" in q or "menor população" in q) else "DESC"
        sql = f"""
SELECT
    municipio_nome,
    uf_sigla,
    ano,
    populacao_estimada
FROM ibge_populacao_municipios
{where_uf}
ORDER BY populacao_estimada {order_dir}
LIMIT {limit_n}
""".strip()
        return sql, extra_queries

    # Se um LLM estiver disponível e nenhum padrão direto bateu, pedir ao LLM para gerar o SQL
    if llm is not None and tables_schema:
        schema_str = json.dumps(tables_schema, ensure_ascii=False, indent=2)
        prompt = (
            "Você é um especialista em DuckDB SQL.\n"
            "Com base nos esquemas de tabelas abaixo, escreva APENAS uma consulta SQL SELECT válida (sem markdown, sem comentários) "
            "que responda à pergunta do usuário.\n\n"
            f"Esquemas disponíveis:\n{schema_str}\n\n"
            f"Pergunta: {question}\n"
            "SQL:"
        )
        llm_sql = invoke_llm_text(llm, prompt)
        if llm_sql:
            clean_sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", llm_sql.strip(), flags=re.IGNORECASE).strip()
            return clean_sql, extra_queries

    # Fallback genérico: listar resumo das UFs
    return "SELECT uf_id, uf_sigla, uf_nome, regiao_nome FROM ibge_ufs ORDER BY uf_sigla LIMIT 27", extra_queries


def evaluate_chunks_relevance(
    question: str,
    chunks: list[dict[str, Any]],
    llm: BaseChatModel | None = None,
) -> bool:
    """Avalia se os trechos recuperados têm qualidade e relevância suficientes para responder."""
    if not chunks:
        return False

    top_score = max(float(c.get("score", 0.0)) for c in chunks)
    # Extrair palavras significativas da pergunta (>= 4 letras)
    stopwords = {
        "qual", "quais", "como", "onde", "quando", "porque", "para", "pela",
        "pelo", "pelos", "pelas", "sobre", "entre", "segundo", "conforme",
        "previsto", "prevista", "prevê", "preve", "estabelece", "relacao",
        "relação", "últimos", "ultimos", "anos", "meses", "disponíveis",
        "prazo", "direito", "dever", "clt", "cdc", "lgpd", "marco", "civil",
        "internet", "lei", "artigo", "disposto", "trata", "fala",
    }
    q_words = [
        w for w in re.findall(r"[a-zà-ú0-9]{4,}", question.lower())
        if w not in stopwords
    ]
    combined_text = " ".join(str(c.get("text", "")).lower() for c in chunks)

    keyword_hits = sum(1 for w in q_words if w in combined_text)

    if q_words and keyword_hits == 0:
        return False

    if top_score < 0.25 and keyword_hits == 0:
        return False

    if llm is not None:
        snippets = "\n---\n".join(str(c.get("text", ""))[:600] for c in chunks[:3])
        prompt = (
            "Avalie se os trechos de documentos abaixo contêm informações relevantes para ajudar a responder à pergunta.\n"
            f"Pergunta: {question}\n\n"
            f"Trechos:\n{snippets}\n\n"
            "Responda APENAS 'SIM' ou 'NAO'."
        )
        verdict = invoke_llm_text(llm, prompt)
        if verdict and "NAO" in verdict.upper().replace("Ã", "A") and top_score < 0.45:
            return False

    return top_score >= 0.28 or keyword_hits >= 1


def synthesize_final_answer(
    state: AgentState,
    llm: BaseChatModel | None = None,
    mcp_prompt_template: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Gera a resposta final em português sempre citando as fontes (lei+artigo e/ou tabela+SQL)."""
    question = state.get("question", "")
    route = state.get("route", "text")
    chunks = state.get("retrieved_chunks") or []
    docs_relevant = state.get("docs_relevant", False)
    sql_results = state.get("sql_results") or {}
    generated_sql = state.get("generated_sql", "")
    sql_error = state.get("sql_error")

    sources: list[dict[str, Any]] = []

    # Coletar fontes documentais se relevantes
    if route in ("text", "mixed") and chunks and docs_relevant:
        seen_doc_sources: set[str] = set()
        for c in chunks[:4]:
            meta = c.get("metadata") or {}
            law_name = meta.get("law_name") or meta.get("filename") or meta.get("doc_id") or "Documento"
            law_number = meta.get("law_number") or ""
            article = meta.get("article") or ""
            key = f"{law_name}|{law_number}|{article}"
            if key not in seen_doc_sources:
                seen_doc_sources.add(key)
                sources.append(
                    {
                        "type": "document",
                        "law_name": law_name,
                        "law_number": law_number,
                        "article": article,
                        "source_url": meta.get("source_url", ""),
                        "score": c.get("score", 0.0),
                        "excerpt": str(c.get("text", ""))[:400],
                    }
                )

    # Coletar fontes de tabelas se houve consulta SQL bem-sucedida
    rows = sql_results.get("rows") or []
    if route in ("table", "mixed") and generated_sql and not sql_error:
        tables_mentioned = re.findall(
            r"\b(bcb_selic_meta|bcb_ipca_mensal|bcb_dolar_ptax_venda|ibge_ufs|ibge_municipios|ibge_populacao_municipios|ibge_pib_municipios|[a-z0-9_]+)\b",
            generated_sql.lower(),
        )
        known_tables = [
            t for t in dict.fromkeys(tables_mentioned)
            if t.startswith(("bcb_", "ibge_", "tbl_")) or t in {"vendas", "clientes", "produtos"}
        ]
        table_label = ", ".join(known_tables) if known_tables else "tabelas_duckdb"
        sources.append(
            {
                "type": "table",
                "table_name": table_label,
                "sql": generated_sql,
                "row_count": len(rows),
            }
        )

    # Verificar se há base suficiente para responder
    has_doc_evidence = bool(route in ("text", "mixed") and chunks and docs_relevant)
    has_table_evidence = bool(route in ("table", "mixed") and rows and not sql_error)

    if not has_doc_evidence and not has_table_evidence:
        no_basis_msg = (
            "Não encontrei informações suficientes nas fontes disponíveis (documentos indexados ou tabelas) "
            "para responder com segurança a essa pergunta. Por favor, verifique se o documento ou tabela "
            "correspondente foi carregado no workspace."
        )
        return no_basis_msg, sources

    # Montar contexto textual para o LLM ou para o sintetizador estruturado
    context_blocks: list[str] = []
    if has_doc_evidence:
        doc_lines = ["### Trechos de Documentos Recuperados:"]
        for idx, c in enumerate(chunks[:4], start=1):
            meta = c.get("metadata") or {}
            cite = f"{meta.get('law_name', '')} ({meta.get('law_number', '')}) - {meta.get('article', '')}".strip(" -()")
            doc_lines.append(f"[{idx}] Fonte: {cite} (Score: {c.get('score')})\n{c.get('text', '')}")
        context_blocks.append("\n\n".join(doc_lines))

    if has_table_evidence:
        table_lines = [
            "### Resultado da Consulta em Tabelas (DuckDB):",
            f"SQL Executado:\n```sql\n{generated_sql}\n```",
            f"Total de linhas retornadas: {len(rows)}",
            f"Dados JSON:\n{json.dumps(rows[:25], ensure_ascii=False, indent=2)}",
        ]
        context_blocks.append("\n\n".join(table_lines))

    full_context = "\n\n".join(context_blocks)

    # Tentar geração via LLM (Claude / Ollama) usando o prompt MCP responder_com_fontes
    if llm is not None:
        prompt_to_send = (
            mcp_prompt_template
            if mcp_prompt_template
            else (
                "Responda em português do Brasil com base exclusivamente no contexto abaixo, "
                "citando sempre as fontes (lei, número e artigo para textos; nome da tabela e consulta SQL para dados numéricos).\n\n"
                f"Contexto:\n{full_context}\n\nPergunta: {question}"
            )
        )
        llm_answer = invoke_llm_text(llm, prompt_to_send)
        if llm_answer:
            # Garantir que a seção de fontes esteja explicitamente anexada ao final
            sources_footer = _format_sources_footer(sources)
            if "Fontes Citadas" not in llm_answer:
                llm_answer = f"{llm_answer.rstrip()}\n\n{sources_footer}"
            return llm_answer, sources

    # Fallback determinístico estruturado em português (garante funcionamento 100% offline/sem API key)
    answer_parts: list[str] = []

    if has_doc_evidence:
        answer_parts.append("### Análise Documental / Legislação")
        for c in chunks[:3]:
            meta = c.get("metadata") or {}
            law_name = meta.get("law_name") or meta.get("filename") or "Documento"
            law_num = meta.get("law_number") or ""
            art = meta.get("article") or ""
            header_cite = f"**{law_name}" + (f" ({law_num})" if law_num else "") + (f", {art}**" if art else "**")
            # Limpar a primeira linha de cabeçalho interno [Lei - Art.] para evitar repetição visual
            raw_lines = [ln for ln in str(c.get("text", "")).splitlines() if not ln.startswith("[") and not ln.startswith("Contexto:")]
            clean_excerpt = "\n".join(raw_lines).strip()
            answer_parts.append(f"- Segundo a {header_cite}:\n  > {clean_excerpt}")

    if has_table_evidence:
        answer_parts.append("### Dados e Indicadores das Tabelas")
        if len(rows) == 1:
            row_obj = rows[0]
            kv_items = [f"**{k}**: `{v}`" for k, v in row_obj.items()]
            answer_parts.append("Resultado apurado na consulta: " + ", ".join(kv_items) + ".")
        else:
            # Formatar tabela Markdown para até 15 linhas
            cols = list(rows[0].keys())
            md_header = "| " + " | ".join(cols) + " |"
            md_sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            md_rows = [
                "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
                for r in rows[:15]
            ]
            answer_parts.append("\n".join([md_header, md_sep, *md_rows]))

        # Se for a pergunta mista sobre relação entre Selic e inflação (IPCA), adicionar síntese analítica dos dados
        q_low = question.lower()
        if "selic" in q_low and ("inflação" in q_low or "inflacao" in q_low or "ipca" in q_low):
            answer_parts.append(
                "\n**Síntese da relação observada nos dados:** A série histórica do Banco Central demonstra que a "
                "taxa Selic meta (`bcb_selic_meta`, código SGS 432) atua como o principal instrumento de política monetária "
                "para controle da inflação medida pelo IPCA (`bcb_ipca_mensal`, código SGS 433), mantendo patamares de juros "
                "reais positivos nos últimos anos analisados para convergir a inflação acumulada em direção à meta."
            )

    answer_parts.append(_format_sources_footer(sources))
    return "\n\n".join(answer_parts), sources


def _format_sources_footer(sources: list[dict[str, Any]]) -> str:
    """Formata o bloco padronizado de Fontes Citadas ao final da resposta."""
    lines = ["---", "### Fontes Citadas"]
    if not sources:
        lines.append("- Nenhuma fonte aplicável encontrada.")
        return "\n".join(lines)

    for src in sources:
        if src.get("type") == "document":
            law = src.get("law_name", "Documento")
            num = src.get("law_number", "")
            art = src.get("article", "")
            url = src.get("source_url", "")
            num_str = f" ({num})" if num else ""
            art_str = f" — **{art}**" if art else ""
            url_str = f" — Fonte: {url}" if url else ""
            lines.append(f"- **Documento/Lei:** {law}{num_str}{art_str}{url_str}")
        elif src.get("type") == "table":
            tbl = src.get("table_name", "tabela")
            sql_one_line = " ".join(str(src.get("sql", "")).split())
            lines.append(f"- **Tabela(s):** `{tbl}`")
            lines.append(f"  - **Consulta SQL executada:** `{sql_one_line}`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Construção do Grafo LangGraph consumindo as ferramentas MCP
# ---------------------------------------------------------------------------

def _unwrap_tool_output(raw_output: Any) -> Any:
    """Converte o retorno de uma BaseTool do langchain-mcp-adapters em dicionário Python."""
    if isinstance(raw_output, dict):
        return raw_output
    if isinstance(raw_output, list):
        # langchain-mcp-adapters pode retornar lista de blocos [{'type': 'text', 'text': '...'}]
        text_parts: list[str] = []
        for item in raw_output:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                text_parts.append(str(item.text))
            else:
                text_parts.append(str(item))
        joined = "\n".join(text_parts).strip()
        try:
            return json.loads(joined)
        except Exception:
            return {"raw": joined}
    if isinstance(raw_output, str):
        try:
            return json.loads(raw_output)
        except Exception:
            return {"raw": raw_output}
    return {"raw": str(raw_output)}


def build_agent_graph(
    mcp_tools: list[BaseTool],
    llm: BaseChatModel | None = None,
    config: AppConfig | None = None,
    session: ClientSession | None = None,
) -> Any:
    """Constrói e compila o StateGraph do LangGraph conectado às ferramentas carregadas via MCP."""
    cfg = config or load_config()
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in mcp_tools}

    required_tools = {"search_docs", "list_tables", "describe_table", "query_table"}
    missing = required_tools - set(tools_by_name.keys())
    if missing:
        raise ValueError(f"Ferramentas MCP obrigatórias ausentes: {missing}")

    # 1. Nó de Classificação (Roteador)
    async def classify_node(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        workspace = state.get("workspace") or cfg.storage.default_workspace
        route = classify_question(question, llm=llm)
        doc_filter = detect_doc_filter_from_question(question)
        logger.info(f"[LangGraph] Pergunta classificada como '{route}' (doc_filter={doc_filter})")
        return {
            "workspace": workspace,
            "route": route,
            "search_query": question,
            "doc_filter": doc_filter,
            "rewrite_count": 0,
            "max_rewrites": cfg.agent.max_rewrite_retries,
        }

    # 2. Nó de Busca em Documentos via MCP tool `search_docs`
    async def search_docs_node(state: AgentState) -> dict[str, Any]:
        query = state.get("search_query") or state["question"]
        workspace = state.get("workspace") or "default"
        doc_filter = state.get("doc_filter")
        top_k = cfg.agent.top_k

        tool_args: dict[str, Any] = {
            "query": query,
            "workspace": workspace,
            "top_k": top_k,
        }
        if doc_filter:
            tool_args["doc_filter"] = doc_filter

        raw_res = await tools_by_name["search_docs"].ainvoke(tool_args)
        parsed = _unwrap_tool_output(raw_res)
        results = parsed.get("results") or []

        # Se o filtro por documento não retornou bons resultados na reescrita, buscar em todos os documentos
        if not results and doc_filter:
            raw_res_all = await tools_by_name["search_docs"].ainvoke(
                {"query": query, "workspace": workspace, "top_k": top_k}
            )
            parsed_all = _unwrap_tool_output(raw_res_all)
            results = parsed_all.get("results") or []

        return {"retrieved_chunks": results}

    # 3. Nó de Avaliação de Relevância dos Trechos
    async def evaluate_relevance_node(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        chunks = state.get("retrieved_chunks") or []
        is_rel = evaluate_chunks_relevance(question=question, chunks=chunks, llm=llm)
        logger.info(
            f"[LangGraph] Avaliação de relevância dos chunks: {is_rel} (tentativa={state.get('rewrite_count', 0)})"
        )
        return {"docs_relevant": is_rel}

    # 4. Nó de Reescrita de Busca (máximo 2 vezes configurável em max_rewrites)
    async def rewrite_query_node(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        current_count = int(state.get("rewrite_count", 0)) + 1

        new_query: str | None = None
        if llm is not None:
            prompt = (
                "A busca semântica anterior não retornou artigos suficientemente relevantes.\n"
                "Reescreva a consulta abaixo adicionando sinônimos jurídicos e termos técnicos "
                "da legislação brasileira (CLT, CDC, LGPD, Marco Civil da Internet) para melhorar a recuperação.\n"
                f"Pergunta original: {question}\n"
                "Responda APENAS com a nova consulta de busca."
            )
            new_query = invoke_llm_text(llm, prompt)

        if not new_query:
            new_query = expand_rewritten_query_deterministic(question, attempt=current_count)

        logger.info(f"[LangGraph] Reescrevendo query ({current_count}/{state.get('max_rewrites', 2)}): {new_query}")
        # Na segunda tentativa, removemos eventual doc_filter restritivo para ampliar a cobertura
        next_filter = state.get("doc_filter") if current_count == 1 else None
        return {
            "search_query": new_query,
            "rewrite_count": current_count,
            "doc_filter": next_filter,
        }

    # 5. Nó de Inspeção de Esquema de Tabelas via MCP tools `list_tables` e `describe_table`
    async def inspect_schema_node(state: AgentState) -> dict[str, Any]:
        workspace = state.get("workspace") or "default"
        raw_list = await tools_by_name["list_tables"].ainvoke({"workspace": workspace})
        parsed_list = _unwrap_tool_output(raw_list)
        tables = parsed_list.get("tables") or []

        q_low = state["question"].lower()
        detailed_schemas: list[dict[str, Any]] = []
        for t in tables:
            tname = t.get("table_name", "")
            # Descrever tabelas relevantes para a pergunta (ou todas se forem poucas)
            should_describe = (
                len(tables) <= 8
                or any(part in q_low for part in tname.split("_") if len(part) > 2)
            )
            if should_describe and tname:
                raw_desc = await tools_by_name["describe_table"].ainvoke(
                    {"name": tname, "workspace": workspace}
                )
                detailed_schemas.append(_unwrap_tool_output(raw_desc))

        return {"tables_schema": detailed_schemas}

    # 6. Nó de Geração de SQL
    async def generate_sql_node(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        schemas = state.get("tables_schema") or []
        sql, _ = build_sql_for_question(question=question, tables_schema=schemas, llm=llm)
        return {"generated_sql": sql}

    # 7. Nó de Validação de SQL
    async def validate_sql_node(state: AgentState) -> dict[str, Any]:
        sql = state.get("generated_sql", "")
        try:
            clean_sql = validate_readonly_sql(sql)
            return {"generated_sql": clean_sql, "sql_valid": True, "sql_error": None}
        except UnsafeSQLError as err:
            logger.warning(f"[LangGraph] SQL bloqueado na validação do agente: {err}")
            return {"sql_valid": False, "sql_error": str(err)}

    # 8. Nó de Execução de SQL via MCP tool `query_table`
    async def execute_sql_node(state: AgentState) -> dict[str, Any]:
        sql = state.get("generated_sql", "")
        workspace = state.get("workspace") or "default"
        raw_res = await tools_by_name["query_table"].ainvoke(
            {"sql": sql, "workspace": workspace}
        )
        parsed = _unwrap_tool_output(raw_res)
        if "error" in parsed:
            return {
                "sql_results": {},
                "sql_error": f"{parsed.get('error')}: {parsed.get('message')}",
            }
        return {"sql_results": parsed, "sql_error": None}

    # 9. Nó de Geração da Resposta Final com Fontes
    async def generate_answer_node(state: AgentState) -> dict[str, Any]:
        mcp_prompt_str: str | None = None
        if session is not None and llm is not None:
            try:
                msgs = await load_mcp_prompt(
                    session,
                    "responder_com_fontes",
                    arguments={"pergunta": state["question"], "contexto": ""},
                )
                if msgs:
                    mcp_prompt_str = str(msgs[0].content)
            except Exception as err:
                logger.debug(f"Aviso ao carregar prompt MCP: {err}")

        answer, sources = synthesize_final_answer(
            state=state,
            llm=llm,
            mcp_prompt_template=mcp_prompt_str,
        )
        return {"answer": answer, "sources": sources}

    # Roteadores condicionais
    def route_after_classify(state: AgentState) -> str:
        route = state.get("route", "text")
        if route == "table":
            return "inspect_schema"
        # Tanto 'text' quanto 'mixed' iniciam buscando nos documentos
        return "search_docs"

    def route_after_evaluate_relevance(state: AgentState) -> str:
        is_rel = state.get("docs_relevant", False)
        rewrite_count = int(state.get("rewrite_count", 0))
        max_rewrites = int(state.get("max_rewrites", 2))

        if not is_rel and rewrite_count < max_rewrites:
            return "rewrite_query"

        # Se a rota for mista ('mixed'), após concluir a etapa de texto seguimos para a etapa de tabelas
        if state.get("route") == "mixed":
            return "inspect_schema"
        return "generate_answer"

    def route_after_validate_sql(state: AgentState) -> str:
        if state.get("sql_valid"):
            return "execute_sql"
        return "generate_answer"

    workflow = StateGraph(AgentState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("search_docs", search_docs_node)
    workflow.add_node("evaluate_relevance", evaluate_relevance_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("inspect_schema", inspect_schema_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "search_docs": "search_docs",
            "inspect_schema": "inspect_schema",
        },
    )
    workflow.add_edge("search_docs", "evaluate_relevance")
    workflow.add_conditional_edges(
        "evaluate_relevance",
        route_after_evaluate_relevance,
        {
            "rewrite_query": "rewrite_query",
            "inspect_schema": "inspect_schema",
            "generate_answer": "generate_answer",
        },
    )
    workflow.add_edge("rewrite_query", "search_docs")
    workflow.add_edge("inspect_schema", "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")
    workflow.add_conditional_edges(
        "validate_sql",
        route_after_validate_sql,
        {
            "execute_sql": "execute_sql",
            "generate_answer": "generate_answer",
        },
    )
    workflow.add_edge("execute_sql", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Sessões MCP via langchain-mcp-adapters (in-memory ou stdio subprocess)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def open_mcp_session(mode: Literal["memory", "stdio"] = "memory") -> AsyncIterator[ClientSession]:
    """Abre uma ClientSession MCP conectada ao nosso servidor FastMCP (em memória ou via stdio)."""
    if mode == "stdio":
        repo_root = Path(__file__).resolve().parent.parent.parent
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server.server", "--transport", "stdio"],
            cwd=str(repo_root),
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
    else:
        async with create_connected_server_and_client_session(fastmcp_server) as session:
            yield session


async def ask_agent_async(
    question: str,
    workspace: str = "default",
    mode: Literal["memory", "stdio"] = "memory",
    llm: BaseChatModel | None = None,
    config: AppConfig | None = None,
) -> AgentState:
    """Executa uma pergunta ponta a ponta no grafo LangGraph conectado ao servidor MCP via langchain-mcp-adapters."""
    cfg = config or load_config()
    active_llm = llm if llm is not None else get_chat_model(cfg)

    async with open_mcp_session(mode=mode) as session:
        mcp_tools = await load_mcp_tools(session)
        app = build_agent_graph(
            mcp_tools=mcp_tools,
            llm=active_llm,
            config=cfg,
            session=session,
        )
        final_state: AgentState = await app.ainvoke(
            {"question": question, "workspace": workspace}
        )
        return final_state


def ask_agent(
    question: str,
    workspace: str = "default",
    mode: Literal["memory", "stdio"] = "memory",
    llm: BaseChatModel | None = None,
    config: AppConfig | None = None,
) -> AgentState:
    """Wrapper síncrono para invocar o agente LangGraph em scripts, CLI ou Streamlit."""
    return asyncio.run(
        ask_agent_async(
            question=question,
            workspace=workspace,
            mode=mode,
            llm=llm,
            config=config,
        )
    )
