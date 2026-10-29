"""Interface web em Streamlit para o assistente docs-chat-mcp."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from src.agent.graph import ask_agent
from src.config import load_config
from src.ingestion.loaders import SUPPORTED_DOC_EXTENSIONS
from src.mcp_server.server import (
    describe_table,
    ingest,
    list_documents,
    list_tables,
)


def _sanitize_ws(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())
    return clean or "default"


def _save_uploaded_files(
    uploaded_files: list[Any],
    workspace: str,
    base_docs_dir: Path,
    base_tables_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Salva arquivos enviados pelo usuário nas pastas do workspace e retorna (docs, csvs)."""
    ws_docs_dir = base_docs_dir if workspace == "default" else (base_docs_dir / workspace)
    ws_tables_dir = base_tables_dir if workspace == "default" else (base_tables_dir / workspace)
    ws_docs_dir.mkdir(parents=True, exist_ok=True)
    ws_tables_dir.mkdir(parents=True, exist_ok=True)

    saved_docs: list[Path] = []
    saved_csvs: list[Path] = []

    for up in uploaded_files:
        suffix = Path(up.name).suffix.lower()
        if suffix == ".csv":
            dest = ws_tables_dir / up.name
            dest.write_bytes(up.getvalue())
            saved_csvs.append(dest)
        elif suffix in SUPPORTED_DOC_EXTENSIONS:
            dest = ws_docs_dir / up.name
            dest.write_bytes(up.getvalue())
            saved_docs.append(dest)

    return saved_docs, saved_csvs


def render_sources_panel(sources: list[dict[str, Any]], route: str) -> None:
    """Exibe painel expansível com as fontes documentais e/ou consultas SQL utilizadas."""
    if not sources:
        return

    with st.expander(f"Fontes consultadas ({len(sources)}) — Rota: `{route}`", expanded=False):
        for idx, src in enumerate(sources, start=1):
            if src.get("type") == "document":
                law = src.get("law_name", "Documento")
                num = src.get("law_number", "")
                art = src.get("article", "")
                score = src.get("score", 0.0)
                url = src.get("source_url", "")
                st.markdown(
                    f"**{idx}. Documento:** {law} {f'({num})' if num else ''} — "
                    f"**{art}** *(Score: {score})*"
                )
                if url:
                    st.caption(f"URL de origem: {url}")
                if src.get("excerpt"):
                    st.code(src["excerpt"], language="text")
            elif src.get("type") == "table":
                st.markdown(
                    f"**{idx}. Tabela(s) DuckDB:** `{src.get('table_name', '')}` "
                    f"*(Linhas retornadas: {src.get('row_count', 0)})*"
                )
                st.code(src.get("sql", ""), language="sql")


def main() -> None:
    st.set_page_config(
        page_title="docs-chat-mcp | Assistente de Documentos e Tabelas",
        page_icon="📚",
        layout="wide",
    )

    cfg = load_config()
    base_docs_dir = Path(cfg.storage.docs_dir)
    base_tables_dir = Path(cfg.storage.tables_dir)

    st.title("📚 docs-chat-mcp — Converse com seus Documentos e Tabelas")
    st.caption(
        "Assistente genérico com **Model Context Protocol (MCP)**, **LangGraph**, **ChromaDB** e **DuckDB**."
    )

    # Barra lateral: Workspace, Upload de Arquivos e Acervo Indexado
    with st.sidebar:
        st.header("⚙️ Configuração & Workspace")
        workspace_input = st.text_input(
            "Workspace ativo",
            value=st.session_state.get("workspace", cfg.storage.default_workspace),
            help="Use 'default' para consultar as leis e tabelas de demonstração ou digite um novo nome para isolar seus documentos.",
        )
        workspace = _sanitize_ws(workspace_input)
        st.session_state["workspace"] = workspace

        st.divider()
        st.subheader("📤 Enviar Documentos ou Tabelas")
        uploaded = st.file_uploader(
            "Adicione arquivos PDF, HTML, Markdown, TXT ou CSV",
            type=["pdf", "txt", "md", "markdown", "html", "htm", "csv"],
            accept_multiple_files=True,
        )
        if st.button("Indexar arquivos enviados", use_container_width=True, disabled=not uploaded):
            with st.spinner("Salvando e indexando arquivos..."):
                docs_saved, csvs_saved = _save_uploaded_files(
                    uploaded_files=uploaded or [],
                    workspace=workspace,
                    base_docs_dir=base_docs_dir,
                    base_tables_dir=base_tables_dir,
                )
                for doc_path in docs_saved:
                    ingest(path=str(doc_path), workspace=workspace)
                st.success(
                    f"Processado com sucesso: {len(docs_saved)} documento(s) indexado(s) e {len(csvs_saved)} tabela(s) CSV registrada(s)."
                )

        if workspace == "default" and base_docs_dir.exists():
            if st.button("🔄 Sincronizar pacote de demonstração", use_container_width=True):
                with st.spinner("Verificando hashes e indexando dados de demonstração..."):
                    res = json.loads(ingest(path=str(base_docs_dir), workspace="default"))
                    st.info(
                        f"Arquivos verificados: {res['files_scanned']} | "
                        f"Novos/atualizados: {res['indexed_count']} | "
                        f"Inalterados: {res['skipped_unchanged_count']}"
                    )

        st.divider()
        st.subheader("📄 Documentos no Workspace")
        docs_info = json.loads(list_documents(workspace=workspace))
        docs_list = docs_info.get("documents") or []
        if not docs_list:
            st.caption("Nenhum documento indexado neste workspace.")
        else:
            for d in docs_list:
                label = d.get("law_name") or d.get("filename") or d.get("doc_id")
                st.markdown(f"- **{label}** (`{d.get('chunks_count', 0)}` trechos)")

        st.subheader("📊 Tabelas Disponíveis (DuckDB)")
        tables_info = json.loads(list_tables(workspace=workspace))
        tables_list = tables_info.get("tables") or []
        if not tables_list:
            st.caption("Nenhuma tabela CSV encontrada.")
        else:
            selected_tbl = st.selectbox(
                "Inspecionar esquema da tabela:",
                options=[t["table_name"] for t in tables_list],
            )
            if selected_tbl:
                desc = json.loads(describe_table(name=selected_tbl, workspace=workspace))
                st.caption(f"{desc.get('description', '')} ({desc.get('total_rows', 0):,} linhas)")
                st.dataframe(desc.get("columns", []), use_container_width=True, hide_index=True)

    # Sugestões rápidas de perguntas de demonstração
    st.markdown("**Exemplos de perguntas que você pode fazer:**")
    example_cols = st.columns(5)
    examples = [
        "Qual o prazo de férias previsto na CLT?",
        "O que a LGPD diz sobre consentimento?",
        "Qual foi o IPCA acumulado nos últimos 12 meses disponíveis?",
        "Quais os 10 municípios mais populosos do Paraná?",
        "Qual a relação entre a Selic e a inflação nos últimos 3 anos?",
    ]
    triggered_question: str | None = None
    for col, ex in zip(example_cols, examples):
        if col.button(ex, use_container_width=True):
            triggered_question = ex

    # Histórico de mensagens no chat
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources_panel(msg["sources"], msg.get("route", "text"))

    user_input = st.chat_input("Digite sua pergunta sobre os documentos ou tabelas...")
    active_question = triggered_question or user_input

    if active_question:
        st.session_state["messages"].append({"role": "user", "content": active_question})
        with st.chat_message("user"):
            st.markdown(active_question)

        with st.chat_message("assistant"):
            with st.spinner("Consultando servidor MCP via LangGraph..."):
                result_state = ask_agent(question=active_question, workspace=workspace)
                answer_text = result_state.get("answer", "")
                sources_list = result_state.get("sources") or []
                route_used = result_state.get("route", "text")

                st.markdown(answer_text)
                render_sources_panel(sources_list, route_used)

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": sources_list,
                "route": route_used,
            }
        )


if __name__ == "__main__":
    main()
