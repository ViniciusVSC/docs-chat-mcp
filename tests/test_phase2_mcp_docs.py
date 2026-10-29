"""Testes automatizados para a Fase 2 (Ingestão, Chunking, Hash incremental e Ferramentas MCP de busca)."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.indexer import DocumentVectorStore
from src.ingestion.loaders import load_document
from src.mcp_server.server import (
    ingest,
    list_documents,
    read_indexed_document_resource,
    responder_com_fontes,
    search_docs,
    set_vector_store,
)


def test_loaders_and_chunking_for_pdf_html_md_and_law(tmp_path: Path) -> None:
    """Verifica leitura de PDF, HTML, Markdown e chunking por artigo para leis."""
    # 1. Criar um PDF de teste com PyMuPDF
    pdf_path = tmp_path / "manual_usuario.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Manual de Operação do Sistema RAG.\nEste manual explica como indexar PDFs.")
    doc.save(pdf_path)
    doc.close()

    loaded_pdf = load_document(pdf_path)
    assert "Manual de Operação" in loaded_pdf.content
    pdf_chunks = chunk_document(loaded_pdf, chunk_size=500, chunk_overlap=50)
    assert len(pdf_chunks) >= 1
    assert pdf_chunks[0].metadata["filename"] == "manual_usuario.pdf"

    # 2. Criar um HTML com tag <strike>
    html_path = tmp_path / "norma.html"
    html_path.write_text(
        "<html><body><strike>Texto revogado</strike><p>Texto vigente da norma.</p></body></html>",
        encoding="utf-8",
    )
    loaded_html = load_document(html_path)
    assert "Texto revogado" not in loaded_html.content
    assert "Texto vigente da norma." in loaded_html.content

    # 3. Criar um arquivo de Lei com múltiplos artigos e frontmatter
    law_path = tmp_path / "lei_exemplo.txt"
    law_path.write_text(
        "---\n"
        "law_id: lei_exemplo\n"
        "lei: Lei de Exemplo\n"
        "numero: Lei nº 1.234/2025\n"
        "---\n\n"
        "## TÍTULO I\n"
        "### CAPÍTULO I\n"
        "Art. 1º O prazo de garantia é de 90 dias.\n"
        "Art. 2º O consumidor tem direito a arrependimento em 7 dias.\n"
        "Art. 3º As cláusulas abusivas são nulas de pleno direito.\n"
        "Art. 4º O fornecedor responde objetivamente pelos danos.\n"
        "Art. 5º Esta lei entra em vigor na data de sua publicação.\n",
        encoding="utf-8",
    )
    loaded_law = load_document(law_path)
    law_chunks = chunk_document(loaded_law)
    assert len(law_chunks) == 5
    assert law_chunks[0].metadata["article"] == "Art. 1"
    assert law_chunks[1].metadata["article"] == "Art. 2"
    assert "Lei de Exemplo (Lei nº 1.234/2025) - Art. 2" in law_chunks[1].text


def test_incremental_hash_ingestion_and_mcp_tools(tmp_path: Path) -> None:
    """Testa ingestão via ferramenta MCP, skip por hash inalterado, reindexação se alterado, resource e prompt."""
    chroma_dir = tmp_path / "chroma_test"
    docs_dir = tmp_path / "docs_test"
    docs_dir.mkdir(parents=True, exist_ok=True)

    doc_file = docs_dir / "politica_ferias.md"
    doc_file.write_text(
        "# Política Interna\n\nO colaborador tem direito a 30 dias corridos de férias após 12 meses.",
        encoding="utf-8",
    )

    store = DocumentVectorStore(chroma_dir=chroma_dir)
    set_vector_store(store)

    try:
        # 1ª Ingestão: deve indexar 1 arquivo
        res1 = json.loads(ingest(path=str(docs_dir), workspace="ws_teste"))
        assert res1["indexed_count"] == 1
        assert res1["skipped_unchanged_count"] == 0

        # 2ª Ingestão sem mudanças: deve pular (skipped_unchanged_count == 1)
        res2 = json.loads(ingest(path=str(docs_dir), workspace="ws_teste"))
        assert res2["indexed_count"] == 0
        assert res2["skipped_unchanged_count"] == 1

        # Alterar o conteúdo do arquivo -> hash muda -> deve reindexar
        doc_file.write_text(
            "# Política Interna Atualizada\n\nO colaborador tem direito a 30 dias de férias e abono pecuniário.",
            encoding="utf-8",
        )
        res3 = json.loads(ingest(path=str(docs_dir), workspace="ws_teste"))
        assert res3["indexed_count"] == 1
        assert res3["skipped_unchanged_count"] == 0

        # Listar documentos via MCP tool
        listed = json.loads(list_documents(workspace="ws_teste"))
        assert listed["document_count"] == 1
        assert listed["documents"][0]["doc_id"] == "politica_ferias"
        assert listed["documents"][0]["resource_uri"] == "docs://ws_teste/politica_ferias"

        # Ler Resource MCP docs://{workspace}/{doc_id}
        resource_text = read_indexed_document_resource(workspace="ws_teste", doc_id="politica_ferias")
        assert "abono pecuniário" in resource_text

        # Buscar via search_docs
        search_res = json.loads(search_docs(query="direito a férias e abono", workspace="ws_teste", top_k=3))
        assert search_res["result_count"] == 1
        assert "abono pecuniário" in search_res["results"][0]["text"]
        assert search_res["results"][0]["score"] > 0.2

        # Verificar Prompt MCP responder_com_fontes
        prompt_text = responder_com_fontes(pergunta="Qual o prazo de férias?", contexto="Art. 130 - 30 dias")
        assert "português do Brasil" in prompt_text
        assert "Art. 130 - 30 dias" in prompt_text
    finally:
        set_vector_store(None)


def test_real_legislation_ingestion_and_semantic_search() -> None:
    """Indexa as 4 leis reais em data/docs no workspace 'default' e testa buscas reais da CLT e LGPD."""
    store = DocumentVectorStore()
    set_vector_store(store)
    try:
        ingest_res = json.loads(ingest(path="data/docs", workspace="default"))
        assert ingest_res["files_scanned"] == 4
        assert ingest_res["indexed_count"] + ingest_res["skipped_unchanged_count"] == 4

        docs_list = json.loads(list_documents(workspace="default"))
        doc_ids = {d["doc_id"] for d in docs_list["documents"]}
        assert {"clt", "cdc", "lgpd", "marco_civil_internet"}.issubset(doc_ids)

        # Busca 1: Prazo de férias na CLT (deve recuperar Art. 130 ou artigos do capítulo de férias da CLT)
        clt_res = json.loads(
            search_docs(
                query="Após cada período de 12 meses de vigência do contrato de trabalho o empregado terá direito a férias na seguinte proporção 30 dias corridos",
                workspace="default",
                top_k=5,
                doc_filter="clt",
            )
        )
        assert clt_res["result_count"] == 5
        articles_found = [r["metadata"]["article"] for r in clt_res["results"]]
        texts_combined = "\n".join(r["text"] for r in clt_res["results"])
        assert "Art. 130" in articles_found or "férias" in texts_combined.lower()

        # Busca 2: Consentimento na LGPD
        lgpd_res = json.loads(
            search_docs(
                query="consentimento pelo titular tratamento de dados pessoais manifestação livre informada e inequívoca",
                workspace="default",
                top_k=5,
                doc_filter="lgpd",
            )
        )
        assert lgpd_res["result_count"] == 5
        lgpd_articles = [r["metadata"]["article"] for r in lgpd_res["results"]]
        assert any(a in lgpd_articles for a in ("Art. 7", "Art. 8", "Art. 5"))
    finally:
        set_vector_store(None)
