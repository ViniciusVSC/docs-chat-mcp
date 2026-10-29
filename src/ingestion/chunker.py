"""Divisão de documentos em chunks (por artigo para legislação e janelas com overlap para textos gerais)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.ingestion.loaders import LoadedDocument

ARTICLE_HEADER_RE = re.compile(
    r"^(Art\.?\s*(\d+(?:-[A-Za-z])?)\s*[º°.]?)",
    re.IGNORECASE,
)
TITLE_HEADER_RE = re.compile(r"^(?:#+\s*)?(T[ÍI]TULO\b.*)", re.IGNORECASE)
CHAPTER_HEADER_RE = re.compile(r"^(?:#+\s*)?(CAP[ÍI]TULO\b.*)", re.IGNORECASE)
SECTION_HEADER_RE = re.compile(r"^(?:#+\s*)?((?:SUB)?SE[ÇC][ÃA]O\b.*)", re.IGNORECASE)


@dataclass
class DocumentChunk:
    """Representa um trecho (chunk) pronto para indexação vetorial."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def split_text_with_overlap(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Divide texto longo em blocos de até chunk_size caracteres com chunk_overlap."""
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks: list[str] = []
    start = 0
    text_len = len(clean)
    step = max(1, chunk_size - chunk_overlap)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Tentar quebrar em fim de linha ou espaço próximo ao limite
            newline_pos = clean.rfind("\n", start + int(chunk_size * 0.5), end)
            space_pos = clean.rfind(" ", start + int(chunk_size * 0.6), end)
            if newline_pos != -1:
                end = newline_pos
            elif space_pos != -1:
                end = space_pos

        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break
        start = max(start + 1, end - chunk_overlap)

    return chunks


def is_legislation_document(doc: LoadedDocument) -> bool:
    """Detecta se o documento é uma lei/norma (via metadados ou frequência de 'Art. N')."""
    if doc.metadata.get("law_name") or doc.metadata.get("law_number"):
        return True
    matches = re.findall(r"(?m)^Art\.?\s*\d+", doc.content)
    return len(matches) >= 5


def chunk_legislation_by_article(
    doc: LoadedDocument,
    max_article_chars: int = 2200,
    overlap_chars: int = 250,
) -> list[DocumentChunk]:
    """Divide uma lei por artigo, preservando o contexto (Lei, Título, Capítulo, Artigo).

    Caso um único artigo seja muito extenso (ex.: Art. 482 ou Art. 5º com dezenas de incisos),
    subdivide o artigo mantendo o cabeçalho contextual em cada sub-chunk.
    """
    current_title = ""
    current_chapter = ""
    current_section = ""

    current_article_label = ""
    current_lines: list[str] = []

    raw_articles: list[tuple[str, str, str, str, str]] = []

    def flush() -> None:
        nonlocal current_article_label, current_lines
        if current_article_label and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                raw_articles.append(
                    (
                        current_article_label,
                        current_title,
                        current_chapter,
                        current_section,
                        body,
                    )
                )
        current_article_label = ""
        current_lines = []

    for raw_line in doc.content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_title = TITLE_HEADER_RE.match(line)
        if m_title:
            flush()
            current_title = m_title.group(1).strip()
            current_chapter = ""
            current_section = ""
            continue

        m_chap = CHAPTER_HEADER_RE.match(line)
        if m_chap:
            flush()
            current_chapter = m_chap.group(1).strip()
            current_section = ""
            continue

        m_sec = SECTION_HEADER_RE.match(line)
        if m_sec:
            flush()
            current_section = m_sec.group(1).strip()
            continue

        m_art = ARTICLE_HEADER_RE.match(line)
        if m_art:
            flush()
            art_num = m_art.group(2)
            current_article_label = f"Art. {art_num}"
            current_lines = [line]
        else:
            if current_article_label:
                current_lines.append(line)

    flush()

    law_name = str(doc.metadata.get("law_name") or doc.doc_id.upper())
    law_number = str(doc.metadata.get("law_number") or "")
    source_url = str(doc.metadata.get("source_url") or "")
    download_date = str(doc.metadata.get("download_date") or "")

    chunks: list[DocumentChunk] = []
    chunk_counter = 0

    for art_label, title, chapter, section, art_body in raw_articles:
        hierarchy_parts = [p for p in (title, chapter, section) if p]
        hierarchy_str = " > ".join(hierarchy_parts)
        header_prefix = f"[{law_name}"
        if law_number:
            header_prefix += f" ({law_number})"
        header_prefix += f" - {art_label}]"
        if hierarchy_str:
            header_prefix += f"\nContexto: {hierarchy_str}"

        sub_bodies = split_text_with_overlap(
            art_body,
            chunk_size=max_article_chars,
            chunk_overlap=overlap_chars,
        )
        for sub_idx, sub_body in enumerate(sub_bodies):
            chunk_counter += 1
            full_chunk_text = f"{header_prefix}\n{sub_body}"
            chunk_id = f"{doc.doc_id}__art_{chunk_counter:05d}"
            meta = {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "law_name": law_name,
                "law_number": law_number,
                "article": art_label,
                "title": title,
                "chapter": chapter,
                "section": section,
                "source_url": source_url,
                "download_date": download_date,
                "chunk_index": chunk_counter,
                "sub_part": sub_idx + 1,
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=full_chunk_text,
                    metadata=meta,
                )
            )

    return chunks


def chunk_generic_document(
    doc: LoadedDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """Divide documentos genéricos (PDF, Markdown, TXT, HTML) usando janela com overlap."""
    pieces = split_text_with_overlap(
        doc.content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[DocumentChunk] = []
    for idx, piece in enumerate(pieces, start=1):
        chunk_id = f"{doc.doc_id}__chunk_{idx:05d}"
        meta = {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "law_name": str(doc.metadata.get("law_name") or doc.filename),
            "law_number": str(doc.metadata.get("law_number") or ""),
            "article": f"Trecho {idx}",
            "title": "",
            "chapter": "",
            "section": "",
            "source_url": str(doc.metadata.get("source_url") or ""),
            "download_date": str(doc.metadata.get("download_date") or ""),
            "chunk_index": idx,
            "sub_part": 1,
        }
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=piece,
                metadata=meta,
            )
        )
    return chunks


def chunk_document(
    doc: LoadedDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """Escolhe a estratégia de chunking adequada (por artigo para leis; com overlap para demais)."""
    if is_legislation_document(doc):
        chunks = chunk_legislation_by_article(doc)
        if chunks:
            return chunks
    return chunk_generic_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
