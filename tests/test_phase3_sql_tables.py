"""Testes automatizados para a Fase 3 (Consulta a tabelas CSV via DuckDB, somente leitura)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.mcp_server.server import (
    describe_table,
    list_tables,
    query_table,
    set_table_engine,
)
from src.mcp_server.sql_engine import (
    TableQueryEngine,
    UnsafeSQLError,
    validate_readonly_sql,
)


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "DROP TABLE ibge_ufs",
        "UPDATE ibge_ufs SET uf_sigla = 'XX' WHERE uf_id = 11",
        "DELETE FROM bcb_ipca_mensal",
        "INSERT INTO ibge_ufs VALUES (99, 'ZZ', 'Teste', 1, 'N', 'Norte')",
        "ALTER TABLE ibge_ufs ADD COLUMN extra INT",
        "CREATE TABLE hack AS SELECT * FROM ibge_ufs",
        "TRUNCATE TABLE ibge_ufs",
        "COPY ibge_ufs TO 'hack.csv'",
        "ATTACH 'other.db' AS other",
        "PRAGMA version",
        "SELECT * FROM ibge_ufs; DROP TABLE ibge_ufs;",
        "SELECT * FROM read_csv_auto('C:/Windows/win.ini')",
        "WITH cte AS (SELECT 1) DELETE FROM ibge_ufs",
        "",
        "   -- apenas um comentario",
    ],
)
def test_validate_readonly_sql_blocks_forbidden_statements(malicious_sql: str) -> None:
    """Garante que tentativas de escrita, DDL, múltiplos statements ou leitura externa são bloqueadas."""
    with pytest.raises(UnsafeSQLError):
        validate_readonly_sql(malicious_sql)


def test_validate_readonly_sql_allows_valid_select_and_cte() -> None:
    """Garante que SELECTs complexos, CTEs (WITH) e strings contendo palavras reservadas são aceitos."""
    q1 = validate_readonly_sql("SELECT municipio_nome, populacao_estimada FROM ibge_populacao_municipios LIMIT 10;")
    assert q1.startswith("SELECT")
    assert not q1.endswith(";")

    # Palavra 'UPDATE' dentro de literal de string não deve causar falso positivo
    q2 = validate_readonly_sql("SELECT 'UPDATE ou DROP como texto literal' AS nota FROM ibge_ufs WHERE uf_sigla = 'PR'")
    assert "UPDATE ou DROP" in q2

    # CTE com agregação
    q3 = validate_readonly_sql(
        """
        WITH ultimos_12 AS (
            SELECT data, valor FROM bcb_ipca_mensal ORDER BY data DESC LIMIT 12
        )
        SELECT ROUND(SUM(valor), 2) AS soma_simples FROM ultimos_12
        """
    )
    assert "WITH ultimos_12" in q3


def test_mcp_table_tools_list_describe_and_real_queries() -> None:
    """Testa list_tables, describe_table e query_table nas tabelas reais do BCB e IBGE."""
    engine = TableQueryEngine()
    set_table_engine(engine)
    try:
        # 1. list_tables
        listed = json.loads(list_tables(workspace="default"))
        assert listed["table_count"] >= 7
        table_names = {t["table_name"] for t in listed["tables"]}
        assert {
            "bcb_selic_meta",
            "bcb_ipca_mensal",
            "bcb_dolar_ptax_venda",
            "ibge_ufs",
            "ibge_municipios",
            "ibge_populacao_municipios",
            "ibge_pib_municipios",
        }.issubset(table_names)

        # 2. describe_table
        desc_pop = json.loads(describe_table(name="ibge_populacao_municipios", workspace="default"))
        assert desc_pop["table_name"] == "ibge_populacao_municipios"
        assert desc_pop["total_rows"] > 5500
        col_names = [c["name"] for c in desc_pop["columns"]]
        assert "municipio_nome" in col_names
        assert "populacao_estimada" in col_names
        assert len(desc_pop["sample_rows"]) == 5

        # 3. Consulta real: 10 municípios mais populosos do Paraná
        sql_pr = """
        SELECT municipio_nome, uf_sigla, populacao_estimada
        FROM ibge_populacao_municipios
        WHERE uf_sigla = 'PR'
        ORDER BY populacao_estimada DESC
        LIMIT 10
        """
        res_pr = json.loads(query_table(sql=sql_pr, workspace="default"))
        assert "error" not in res_pr
        assert res_pr["row_count"] == 10
        assert res_pr["rows"][0]["municipio_nome"] == "Curitiba"
        assert res_pr["rows"][1]["municipio_nome"] == "Londrina"

        # 4. Consulta real: IPCA acumulado nos últimos 12 meses disponíveis (fórmula composta e soma)
        sql_ipca = """
        WITH ultimos_12 AS (
            SELECT data, valor
            FROM bcb_ipca_mensal
            ORDER BY data DESC
            LIMIT 12
        )
        SELECT
            MIN(data) AS data_inicial,
            MAX(data) AS data_final,
            COUNT(*) AS meses,
            ROUND((EXP(SUM(LN(1.0 + valor / 100.0))) - 1.0) * 100.0, 2) AS ipca_acumulado_composto_pct,
            ROUND(SUM(valor), 2) AS ipca_soma_simples_pct
        FROM ultimos_12
        """
        res_ipca = json.loads(query_table(sql=sql_ipca, workspace="default"))
        assert "error" not in res_ipca
        assert res_ipca["row_count"] == 1
        assert res_ipca["rows"][0]["meses"] == 12
        assert res_ipca["rows"][0]["ipca_acumulado_composto_pct"] is not None

        # 5. Garantir que a tool MCP query_table retorna erro estruturado quando recebe DROP/UPDATE
        blocked_res = json.loads(query_table(sql="DROP TABLE ibge_ufs", workspace="default"))
        assert blocked_res["error"] == "UnsafeSQLError"
        assert "bloqueado" in blocked_res["message"].lower()

        # 6. Testar truncamento por max_rows
        res_limit = engine.query_table("SELECT * FROM ibge_municipios", max_rows=20)
        assert res_limit["row_count"] == 20
        assert res_limit["truncated"] is True
    finally:
        set_table_engine(None)
