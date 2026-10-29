"""Motor de consultas SQL somente-leitura sobre arquivos CSV usando DuckDB."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from src.config import AppConfig, load_config

logger = logging.getLogger(__name__)

FORBIDDEN_SQL_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "GRANT",
    "REVOKE",
    "ATTACH",
    "DETACH",
    "COPY",
    "EXPORT",
    "IMPORT",
    "PRAGMA",
    "SET",
    "RESET",
    "CALL",
    "EXECUTE",
    "PREPARE",
    "VACUUM",
    "LOAD",
    "INSTALL",
    "FORCE",
    "CHECKPOINT",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "MERGE",
}

FORBIDDEN_FUNCTIONS_RE = re.compile(
    r"\b(read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|read_text|"
    r"read_blob|glob|parquet_scan|csv_scan|sqlite_scan|postgres_scan)\s*\(",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    """Exceção levantada quando uma consulta SQL viola a política de somente leitura."""


class SQLTimeoutError(TimeoutError):
    """Exceção levantada quando uma consulta SQL excede o tempo limite configurado."""


def _strip_sql_comments_and_strings(sql: str) -> tuple[str, str]:
    """Remove comentários SQL e produz uma versão com literais de string mascarados para inspeção léxica."""
    # Remover comentários de bloco /* ... */ e de linha -- ...
    no_comments = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_comments = re.sub(r"--[^\r\n]*", " ", no_comments)
    no_comments = no_comments.strip()

    # Mascarar strings simples '...' e identificadores entre aspas duplas "..."
    masked = re.sub(r"'(?:''|[^'])*'", "''", no_comments)
    masked = re.sub(r'"(?:""|[^"])*"', '""', masked)
    return no_comments, masked


def validate_readonly_sql(sql: str) -> str:
    """Valida rigorosamente se a consulta SQL é apenas de leitura (SELECT / WITH ... SELECT).

    Retorna a query limpa (sem ponto e vírgula final) ou levanta UnsafeSQLError.
    """
    if not sql or not sql.strip():
        raise UnsafeSQLError("A consulta SQL não pode ser vazia.")

    cleaned, masked = _strip_sql_comments_and_strings(sql)
    if not cleaned:
        raise UnsafeSQLError("A consulta SQL não contém instruções válidas.")

    # Remover no máximo um ';' no final absoluto e proibir ';' intermediários (múltiplos statements)
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    while masked.endswith(";"):
        masked = masked[:-1].strip()

    if ";" in masked:
        raise UnsafeSQLError(
            "Múltiplos comandos SQL (separados por ';') não são permitidos por motivos de segurança."
        )

    # Deve iniciar com SELECT ou WITH
    first_token_match = re.match(r"^([A-Za-z_]+)", masked)
    if not first_token_match:
        raise UnsafeSQLError("A consulta SQL deve iniciar com SELECT ou WITH.")

    first_keyword = first_token_match.group(1).upper()
    if first_keyword not in {"SELECT", "WITH"}:
        raise UnsafeSQLError(
            f"Apenas consultas SELECT são permitidas. Comando '{first_keyword}' bloqueado."
        )

    # Verificar palavras-chave proibidas no texto mascarado
    tokens = {t.upper() for t in re.findall(r"\b[A-Za-z_]+\b", masked)}
    forbidden_found = tokens.intersection(FORBIDDEN_SQL_KEYWORDS)
    if forbidden_found:
        sorted_found = ", ".join(sorted(forbidden_found))
        raise UnsafeSQLError(
            f"Operação de escrita ou administração detectada e bloqueada: {sorted_found}."
        )

    # Bloquear funções de leitura de arquivos arbitrários do sistema
    func_match = FORBIDDEN_FUNCTIONS_RE.search(masked)
    if func_match:
        raise UnsafeSQLError(
            f"Função de leitura direta de arquivo '{func_match.group(1)}' não é permitida. "
            "Consulte apenas as views das tabelas registradas."
        )

    return cleaned


def _serialize_value(val: Any) -> Any:
    """Converte tipos especiais retornados pelo DuckDB (date, datetime, Decimal) para JSON."""
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    return val


def sanitize_table_name(stem: str) -> str:
    """Normaliza o nome de arquivo CSV para um identificador SQL seguro em snake_case."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", stem.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean or clean[0].isdigit():
        clean = f"tbl_{clean}"
    return clean


class TableQueryEngine:
    """Gerencia o registro de CSVs no DuckDB e execução de consultas somente leitura."""

    def __init__(
        self,
        config: AppConfig | None = None,
        tables_dir: str | Path | None = None,
    ) -> None:
        self.config = config or load_config()
        self.base_tables_dir = Path(tables_dir or self.config.storage.tables_dir)
        self.base_tables_dir.mkdir(parents=True, exist_ok=True)
        self.max_rows = self.config.duckdb.max_rows
        self.timeout_seconds = self.config.duckdb.timeout_seconds

    def _resolve_csv_files(self, workspace: str = "default") -> dict[str, Path]:
        """Localiza todos os arquivos CSV disponíveis na pasta padrão e/ou na pasta do workspace."""
        csv_map: dict[str, Path] = {}

        # 1. Tabelas globais em base_tables_dir (ex: data/tables/*.csv)
        if self.base_tables_dir.exists():
            for p in sorted(self.base_tables_dir.glob("*.csv")):
                tname = sanitize_table_name(p.stem)
                csv_map[tname] = p.resolve()

        # 2. Se o workspace tiver subpasta própria (ex: data/tables/<workspace>/*.csv), inclui/sobrescreve
        if workspace and workspace != "default":
            ws_dir = self.base_tables_dir / re.sub(r"[^a-zA-Z0-9_-]", "_", workspace)
            if ws_dir.exists() and ws_dir.is_dir():
                for p in sorted(ws_dir.glob("*.csv")):
                    tname = sanitize_table_name(p.stem)
                    csv_map[tname] = p.resolve()

        return csv_map

    def _load_catalog_metadata(self, workspace: str = "default") -> dict[str, dict[str, Any]]:
        """Carrega descrições amigáveis do catalog.json se existir."""
        catalog_by_name: dict[str, dict[str, Any]] = {}
        candidates = [self.base_tables_dir / "catalog.json"]
        if workspace and workspace != "default":
            candidates.append(self.base_tables_dir / workspace / "catalog.json")

        for cat_path in candidates:
            if cat_path.exists():
                try:
                    entries = json.loads(cat_path.read_text(encoding="utf-8"))
                    if isinstance(entries, list):
                        for item in entries:
                            if isinstance(item, dict) and "table_name" in item:
                                catalog_by_name[item["table_name"]] = item
                except Exception as err:
                    logger.warning(f"Erro ao ler catálogo {cat_path}: {err}")
        return catalog_by_name

    def _create_connection(self, workspace: str = "default") -> tuple[duckdb.DuckDBPyConnection, dict[str, Path]]:
        """Cria conexão DuckDB em memória, carrega as tabelas CSV e trava acesso externo ao sistema de arquivos."""
        conn = duckdb.connect(database=":memory:")
        csv_map = self._resolve_csv_files(workspace=workspace)

        for table_name, csv_path in csv_map.items():
            escaped_path = str(csv_path).replace("'", "''")
            conn.execute(
                f'CREATE TABLE "{table_name}" AS SELECT * FROM read_csv_auto(\'{escaped_path}\', header=true);'
            )

        # Desabilitar acesso ao sistema de arquivos/rede após o carregamento das tabelas em memória
        try:
            conn.execute("SET enable_external_access = false;")
        except Exception as err:
            logger.debug(f"Aviso ao configurar enable_external_access: {err}")

        return conn, csv_map

    def list_tables(self, workspace: str = "default") -> list[dict[str, Any]]:
        """Lista todas as tabelas CSV disponíveis no workspace com contagem de linhas e descrição."""
        csv_map = self._resolve_csv_files(workspace=workspace)
        catalog = self._load_catalog_metadata(workspace=workspace)
        tables_info: list[dict[str, Any]] = []

        for table_name, csv_path in sorted(csv_map.items()):
            cat_entry = catalog.get(table_name, {})
            # Contar linhas rapidamente caso não esteja no catálogo
            row_count = cat_entry.get("row_count")
            if row_count is None:
                try:
                    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                        row_count = max(0, sum(1 for _ in f) - 1)
                except Exception:
                    row_count = 0

            tables_info.append(
                {
                    "table_name": table_name,
                    "file": csv_path.name,
                    "source": cat_entry.get("source", "CSV Local"),
                    "description": cat_entry.get(
                        "description", f"Tabela carregada do arquivo {csv_path.name}"
                    ),
                    "row_count": row_count,
                    "columns_summary": list((cat_entry.get("columns") or {}).keys()),
                }
            )

        return tables_info

    def describe_table(self, name: str, workspace: str = "default") -> dict[str, Any]:
        """Retorna esquema detalhado (colunas, tipos DuckDB, descrições) e amostra de linhas."""
        clean_name = sanitize_table_name(name.removesuffix(".csv"))
        conn, csv_map = self._create_connection(workspace=workspace)
        try:
            if clean_name not in csv_map:
                available = ", ".join(sorted(csv_map.keys()))
                raise KeyError(
                    f"Tabela '{name}' não encontrada. Tabelas disponíveis: {available}"
                )

            catalog = self._load_catalog_metadata(workspace=workspace)
            cat_entry = catalog.get(clean_name, {})
            col_descriptions: dict[str, str] = cat_entry.get("columns") or {}

            desc_rows = conn.execute(f'DESCRIBE "{clean_name}"').fetchall()
            columns_info: list[dict[str, str]] = []
            for col_row in desc_rows:
                col_name = str(col_row[0])
                col_type = str(col_row[1])
                columns_info.append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "description": col_descriptions.get(col_name, ""),
                    }
                )

            count_row = conn.execute(f'SELECT COUNT(*) FROM "{clean_name}"').fetchone()
            total_rows = int(count_row[0]) if count_row else 0

            sample_rel = conn.execute(f'SELECT * FROM "{clean_name}" LIMIT 5')
            sample_cols = [d[0] for d in sample_rel.description]
            sample_rows = [
                dict(zip(sample_cols, (_serialize_value(v) for v in row)))
                for row in sample_rel.fetchall()
            ]

            return {
                "table_name": clean_name,
                "file": csv_map[clean_name].name,
                "source": cat_entry.get("source", "CSV Local"),
                "description": cat_entry.get("description", ""),
                "total_rows": total_rows,
                "columns": columns_info,
                "sample_rows": sample_rows,
            }
        finally:
            conn.close()

    def query_table(
        self,
        sql: str,
        workspace: str = "default",
        max_rows: int | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Valida e executa uma consulta SQL SELECT somente-leitura com timeout e limite de linhas."""
        validated_sql = validate_readonly_sql(sql)
        effective_max_rows = max_rows or self.max_rows
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds

        # Envelopar a query original para impor limite máximo de linhas com segurança
        wrapped_sql = f"SELECT * FROM ({validated_sql}) AS _q LIMIT {effective_max_rows + 1}"

        conn, _ = self._create_connection(workspace=workspace)
        timed_out = False

        def _interrupt_on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            try:
                conn.interrupt()
            except Exception:
                pass

        timer = threading.Timer(effective_timeout, _interrupt_on_timeout)
        timer.daemon = True
        timer.start()

        try:
            rel = conn.execute(wrapped_sql)
            col_names = [desc[0] for desc in rel.description]
            raw_rows = rel.fetchall()
        except Exception as err:
            if timed_out or "interrupt" in str(err).lower():
                raise SQLTimeoutError(
                    f"A consulta SQL excedeu o tempo limite de {effective_timeout}s."
                ) from err
            raise
        finally:
            timer.cancel()
            conn.close()

        truncated = len(raw_rows) > effective_max_rows
        rows_slice = raw_rows[:effective_max_rows]
        serialized_rows = [
            dict(zip(col_names, (_serialize_value(v) for v in row)))
            for row in rows_slice
        ]

        return {
            "sql_executed": validated_sql,
            "columns": col_names,
            "row_count": len(serialized_rows),
            "truncated": truncated,
            "max_rows_limit": effective_max_rows,
            "rows": serialized_rows,
        }
