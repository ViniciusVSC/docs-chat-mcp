"""Servidor MCP (FastMCP) do projeto docs-chat-mcp.

Expõe ferramentas para ingestão e busca semântica em documentos (Fase 2)
e consulta somente-leitura a tabelas via DuckDB (Fase 3), além de
Resources (`docs://{workspace}/{doc_id}`) e Prompt (`responder_com_fontes`).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.config import load_config
from src.ingestion.indexer import DocumentVectorStore
from src.mcp_server.sql_engine import (
    SQLTimeoutError,
    TableQueryEngine,
    UnsafeSQLError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mcp_server")

mcp = FastMCP(
    "docs-chat-mcp",
    instructions=(
        "Assistente MCP para busca semântica em documentos (leis, PDFs, textos) "
        "e consulta analítica em tabelas CSV via SQL somente leitura."
    ),
)

_VECTOR_STORE: DocumentVectorStore | None = None
_TABLE_ENGINE: TableQueryEngine | None = None


def get_vector_store() -> DocumentVectorStore:
    """Retorna a instância singleton do DocumentVectorStore."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        cfg = load_config()
        _VECTOR_STORE = DocumentVectorStore(config=cfg)
    return _VECTOR_STORE


def set_vector_store(store: DocumentVectorStore | None) -> None:
    """Permite injetar um DocumentVectorStore customizado (útil para testes)."""
    global _VECTOR_STORE
    _VECTOR_STORE = store


def get_table_engine() -> TableQueryEngine:
    """Retorna a instância singleton do TableQueryEngine."""
    global _TABLE_ENGINE
    if _TABLE_ENGINE is None:
        cfg = load_config()
        _TABLE_ENGINE = TableQueryEngine(config=cfg)
    return _TABLE_ENGINE


def set_table_engine(engine: TableQueryEngine | None) -> None:
    """Permite injetar um TableQueryEngine customizado (útil para testes)."""
    global _TABLE_ENGINE
    _TABLE_ENGINE = engine


# ---------------------------------------------------------------------------
# Ferramentas MCP de Documentos (Fase 2)
# ---------------------------------------------------------------------------

@mcp.tool()
def ingest(path: str, workspace: str = "default") -> str:
    """Lê, divide em chunks (por artigo para leis; com overlap para textos gerais) e indexa documentos.

    Reindexa apenas arquivos que foram adicionados ou alterados (verificação por hash SHA-256).
    Aceita o caminho de um arquivo individual (.txt, .md, .html, .pdf) ou de um diretório.

    Args:
        path: Caminho do arquivo ou pasta contendo os documentos a indexar (ex: 'data/docs').
        workspace: Nome do workspace/coleção onde os documentos serão organizados (padrão: 'default').
    """
    store = get_vector_store()
    result = store.ingest_path(path=Path(path), workspace=workspace)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def list_documents(workspace: str = "default") -> str:
    """Lista todos os documentos indexados em um workspace, com metadados, número de chunks e URI do resource.

    Use esta ferramenta para descobrir quais leis, manuais ou arquivos estão disponíveis para busca.

    Args:
        workspace: Nome do workspace a consultar (padrão: 'default').
    """
    store = get_vector_store()
    docs = store.list_documents(workspace=workspace)
    enriched: list[dict[str, Any]] = []
    for d in docs:
        item = dict(d)
        item["resource_uri"] = f"docs://{workspace}/{d['doc_id']}"
        enriched.append(item)
    return json.dumps(
        {
            "workspace": workspace,
            "document_count": len(enriched),
            "documents": enriched,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def search_docs(
    query: str,
    workspace: str = "default",
    top_k: int = 5,
    doc_filter: str | None = None,
) -> str:
    """Realiza busca semântica nos documentos indexados e retorna os trechos mais relevantes com metadados e score.

    Use esta ferramenta para responder perguntas conceituais, jurídicas ou textuais (ex: artigos da CLT, CDC, LGPD,
    Marco Civil da Internet ou documentos enviados pelo usuário).

    Args:
        query: Consulta em linguagem natural descrevendo o tema ou dispositivo procurado.
        workspace: Workspace onde buscar (padrão: 'default').
        top_k: Quantidade máxima de trechos retornados (padrão: 5).
        doc_filter: Filtro opcional pelo ID ou nome do documento (ex: 'clt', 'lgpd', 'cdc', 'marco_civil_internet').
    """
    store = get_vector_store()
    results = store.search(
        query=query,
        workspace=workspace,
        top_k=top_k,
        doc_filter=doc_filter,
    )
    return json.dumps(
        {
            "query": query,
            "workspace": workspace,
            "doc_filter": doc_filter,
            "result_count": len(results),
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Ferramentas MCP de Tabelas via DuckDB (Fase 3)
# ---------------------------------------------------------------------------

@mcp.tool()
def list_tables(workspace: str = "default") -> str:
    """Lista todas as tabelas (arquivos CSV) disponíveis para consulta SQL via DuckDB no workspace.

    Retorna o nome exato de cada tabela, descrição da série/conjunto de dados, origem (ex: Banco Central, IBGE),
    quantidade de linhas e resumo das colunas. Use sempre antes de gerar uma query SQL.

    Args:
        workspace: Nome do workspace (padrão: 'default').
    """
    engine = get_table_engine()
    tables = engine.list_tables(workspace=workspace)
    return json.dumps(
        {
            "workspace": workspace,
            "table_count": len(tables),
            "tables": tables,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def describe_table(name: str, workspace: str = "default") -> str:
    """Inspeciona o esquema detalhado de uma tabela (nomes e tipos das colunas no DuckDB, descrições e 5 linhas de amostra).

    Use esta ferramenta antes de escrever consultas SQL para conhecer os tipos exatos das colunas
    (ex: DATE, DOUBLE, BIGINT, VARCHAR) e ver exemplos reais de valores.

    Args:
        name: Nome da tabela (ex: 'bcb_ipca_mensal', 'bcb_selic_meta', 'ibge_populacao_municipios').
        workspace: Nome do workspace (padrão: 'default').
    """
    engine = get_table_engine()
    info = engine.describe_table(name=name, workspace=workspace)
    return json.dumps(info, ensure_ascii=False, indent=2)


@mcp.tool()
def query_table(sql: str, workspace: str = "default") -> str:
    """Executa uma consulta SQL somente leitura (apenas SELECT / WITH) sobre as tabelas CSV no DuckDB.

    Qualquer comando que tente modificar dados ou estruturas (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, ATTACH)
    ou executar múltiplos statements separados por ';' será bloqueado imediatamente.
    A consulta possui timeout e limite máximo de linhas retornadas.

    Args:
        sql: Consulta SQL ANSI/DuckDB iniciando obrigatoriamente por SELECT ou WITH.
        workspace: Nome do workspace onde as tabelas estão registradas (padrão: 'default').
    """
    engine = get_table_engine()
    try:
        result = engine.query_table(sql=sql, workspace=workspace)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (UnsafeSQLError, SQLTimeoutError) as sec_err:
        return json.dumps(
            {
                "error": type(sec_err).__name__,
                "message": str(sec_err),
                "sql_rejected": sql,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as db_err:
        return json.dumps(
            {
                "error": "DuckDBExecutionError",
                "message": str(db_err),
                "sql_failed": sql,
            },
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Resources MCP (docs://{workspace}/{doc_id})
# ---------------------------------------------------------------------------

@mcp.resource(
    "docs://{workspace}/{doc_id}",
    name="documento_indexado",
    description="Retorna o conteúdo textual completo e metadados de um documento indexado no workspace.",
    mime_type="text/plain",
)
def read_indexed_document_resource(workspace: str, doc_id: str) -> str:
    """Resource MCP que permite ler o conteúdo integral de um documento via URI docs://{workspace}/{doc_id}."""
    store = get_vector_store()
    return store.get_document_content(workspace=workspace, doc_id=doc_id)


# ---------------------------------------------------------------------------
# Prompt MCP ("responder_com_fontes")
# ---------------------------------------------------------------------------

@mcp.prompt(
    name="responder_com_fontes",
    description="Instruções rigorosas para responder perguntas em português citando sempre as fontes (lei/artigo ou tabela/SQL).",
)
def responder_com_fontes(pergunta: str, contexto: str = "") -> str:
    """Gera o prompt estruturado exigindo resposta fundamentada e citação explícita das fontes."""
    base_instructions = (
        "Você é um assistente especialista em análise documental e de dados.\n"
        "Regras obrigatórias para sua resposta:\n"
        "1. Responda sempre em português do Brasil, de forma clara, precisa e objetiva.\n"
        "2. Baseie-se EXCLUSIVAMENTE nos trechos de documentos ou nos resultados de consultas SQL fornecidos.\n"
        "3. Cite SEMPRE as fontes ao final ou ao longo do texto:\n"
        "   - Para documentos/leis: informe o nome da lei/documento, o número da norma e o artigo exato (ex: CLT, Decreto-Lei nº 5.452/1943, Art. 130).\n"
        "   - Para tabelas/dados numéricos: informe o nome da tabela e a consulta SQL exata que foi executada.\n"
        "4. Se os dados retornados pelas ferramentas não contiverem base suficiente para responder à pergunta, "
        "diga explicitamente que não encontrou a informação nas fontes disponíveis e NÃO invente números ou artigos.\n"
    )
    if contexto.strip():
        return f"{base_instructions}\nContexto recuperado:\n{contexto}\n\nPergunta do usuário: {pergunta}"
    return f"{base_instructions}\nPergunta do usuário: {pergunta}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inicia o servidor MCP docs-chat-mcp.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Tipo de transporte MCP (padrão: stdio).",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
