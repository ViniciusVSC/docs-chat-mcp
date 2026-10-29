"""Leitores de documentos (TXT, Markdown, HTML, PDF) e extração de metadados."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from bs4 import BeautifulSoup

SUPPORTED_DOC_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


@dataclass
class LoadedDocument:
    """Representa um documento lido do disco antes do chunking."""

    doc_id: str
    filename: str
    path: str
    file_hash: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_file_hash(file_path: Path) -> str:
    """Calcula o hash SHA-256 dos bytes de um arquivo."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extrai cabeçalho YAML simples (--- ... ---) no início do texto, se presente."""
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, text

    end_idx = normalized.find("\n---\n", 4)
    if end_idx == -1:
        return {}, text

    header_block = normalized[4:end_idx]
    body = normalized[end_idx + 5 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in header_block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def read_text_file(file_path: Path) -> str:
    """Lê arquivo TXT ou Markdown tentando UTF-8 e fallback para CP1252."""
    raw = file_path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def read_html_file(file_path: Path) -> str:
    """Lê arquivo HTML, remove tags revogadas (<strike>, <del>, <s>), scripts e estilos."""
    raw = file_path.read_bytes()
    try:
        html_str = raw.decode("utf-8")
    except UnicodeDecodeError:
        html_str = raw.decode("cp1252", errors="replace")

    soup = BeautifulSoup(html_str, "html.parser")
    for tag in soup.find_all(["script", "style", "strike", "s", "del", "head"]):
        tag.decompose()

    lines: list[str] = []
    for raw_line in soup.get_text(separator="\n").splitlines():
        clean = re.sub(r"[ \t\xa0]+", " ", raw_line).strip()
        if clean:
            lines.append(clean)
    return "\n".join(lines)


def read_pdf_file(file_path: Path) -> str:
    """Extrai o texto de todas as páginas de um arquivo PDF via PyMuPDF."""
    pages_text: list[str] = []
    with pymupdf.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages_text.append(f"[Página {page_num}]\n{text}")
    return "\n\n".join(pages_text)


def load_document(file_path: Path) -> LoadedDocument:
    """Carrega um arquivo suportado (TXT, MD, HTML, PDF) e retorna um LoadedDocument."""
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_DOC_EXTENSIONS:
        raise ValueError(
            f"Extensão '{suffix}' não suportada para ingestão de texto ({file_path.name})."
        )

    file_hash = compute_file_hash(file_path)
    doc_id = file_path.stem.lower()

    if suffix in {".txt", ".md", ".markdown"}:
        raw_text = read_text_file(file_path)
    elif suffix in {".html", ".htm"}:
        raw_text = read_html_file(file_path)
    elif suffix == ".pdf":
        raw_text = read_pdf_file(file_path)
    else:
        raw_text = read_text_file(file_path)

    frontmatter, body = parse_frontmatter(raw_text)
    if frontmatter.get("law_id"):
        doc_id = frontmatter["law_id"]

    metadata: dict[str, Any] = {
        "doc_id": doc_id,
        "filename": file_path.name,
        "file_type": suffix.lstrip("."),
        "law_name": frontmatter.get("lei", ""),
        "law_number": frontmatter.get("numero", ""),
        "source_url": frontmatter.get("url_origem", ""),
        "download_date": frontmatter.get("data_download", ""),
    }

    return LoadedDocument(
        doc_id=doc_id,
        filename=file_path.name,
        path=str(file_path.resolve()),
        file_hash=file_hash,
        content=body,
        metadata=metadata,
    )
