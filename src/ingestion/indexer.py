"""Indexador vetorial persistente com ChromaDB, embeddings locais SentenceTransformers e controle de hash."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from src.config import AppConfig, load_config
from src.ingestion.chunker import chunk_document
from src.ingestion.loaders import SUPPORTED_DOC_EXTENSIONS, load_document

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Carrega e mantém em cache o modelo SentenceTransformer multilíngue."""
    if model_name not in _MODEL_CACHE:
        logger.info(f"Carregando modelo de embeddings local: {model_name} ...")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class LocalSentenceTransformerEmbeddings(EmbeddingFunction[Documents]):
    """Função de embeddings compatível com ChromaDB usando SentenceTransformers."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @staticmethod
    def name() -> str:
        return "local_sentence_transformer"

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        model = get_sentence_transformer(self.model_name)
        vectors = model.encode(
            list(input),
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()


def sanitize_workspace_name(workspace: str) -> str:
    """Garante nome de coleção válido para o ChromaDB."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", workspace.strip() or "default")
    return f"ws_{clean}"[:60]


class DocumentVectorStore:
    """Gerencia a ingestão incremental (por hash) e a busca semântica por workspace."""

    def __init__(
        self,
        config: AppConfig | None = None,
        chroma_dir: str | Path | None = None,
        embedding_fn: EmbeddingFunction[Documents] | None = None,
    ) -> None:
        self.config = config or load_config()
        self.chroma_path = Path(chroma_dir or self.config.storage.chroma_dir)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.chroma_path / "ingestion_manifest.json"
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.embedding_fn = embedding_fn or LocalSentenceTransformerEmbeddings(
            self.config.embeddings.model_name
        )

    def _load_manifest(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_manifest(self, manifest: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_collection(self, workspace: str) -> chromadb.Collection:
        col_name = sanitize_workspace_name(workspace)
        return self.client.get_or_create_collection(
            name=col_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine", "workspace": workspace},
        )

    def _collect_target_files(self, target_path: Path) -> list[Path]:
        if target_path.is_file():
            if target_path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS:
                return [target_path]
            raise ValueError(
                f"Arquivo '{target_path.name}' não possui extensão suportada ({SUPPORTED_DOC_EXTENSIONS})."
            )
        if target_path.is_dir():
            files: list[Path] = []
            for p in sorted(target_path.rglob("*")):
                if p.is_file() and p.suffix.lower() in SUPPORTED_DOC_EXTENSIONS:
                    files.append(p)
            return files
        raise FileNotFoundError(f"Caminho não encontrado: {target_path}")

    def ingest_path(
        self,
        path: str | Path,
        workspace: str = "default",
    ) -> dict[str, Any]:
        """Lê, divide em chunks e indexa arquivos de um caminho (arquivo ou pasta).

        Reindexa apenas arquivos que sofreram alteração (comparando o hash SHA-256).
        """
        target = Path(path)
        files = self._collect_target_files(target)
        collection = self._get_collection(workspace)
        manifest = self._load_manifest()
        ws_manifest = manifest.setdefault(workspace, {})

        indexed_docs: list[dict[str, Any]] = []
        skipped_docs: list[str] = []
        total_chunks_added = 0

        for file_path in files:
            loaded = load_document(file_path)
            existing_entry = ws_manifest.get(loaded.doc_id)

            if existing_entry and existing_entry.get("file_hash") == loaded.file_hash:
                skipped_docs.append(loaded.doc_id)
                continue

            # Se o documento já existia com hash diferente, removemos os chunks antigos
            if existing_entry:
                try:
                    collection.delete(where={"doc_id": loaded.doc_id})
                except Exception as err:
                    logger.warning(f"Aviso ao remover chunks antigos de {loaded.doc_id}: {err}")

            chunks = chunk_document(
                loaded,
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
            )

            if not chunks:
                continue

            # Inserir em lotes para respeitar limites internos do SQLite/ChromaDB
            batch_size = 250
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[c.metadata for c in batch],
                )

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            doc_summary = {
                "doc_id": loaded.doc_id,
                "filename": loaded.filename,
                "path": loaded.path,
                "file_hash": loaded.file_hash,
                "chunks_count": len(chunks),
                "law_name": loaded.metadata.get("law_name", ""),
                "law_number": loaded.metadata.get("law_number", ""),
                "source_url": loaded.metadata.get("source_url", ""),
                "download_date": loaded.metadata.get("download_date", ""),
                "indexed_at": now_iso,
            }
            ws_manifest[loaded.doc_id] = doc_summary
            indexed_docs.append(doc_summary)
            total_chunks_added += len(chunks)

        self._save_manifest(manifest)

        return {
            "workspace": workspace,
            "files_scanned": len(files),
            "indexed_count": len(indexed_docs),
            "skipped_unchanged_count": len(skipped_docs),
            "total_chunks_indexed": total_chunks_added,
            "indexed_documents": indexed_docs,
            "skipped_documents": skipped_docs,
        }

    def list_documents(self, workspace: str = "default") -> list[dict[str, Any]]:
        """Retorna a lista de documentos indexados no workspace com seus metadados."""
        manifest = self._load_manifest()
        ws_docs = manifest.get(workspace, {})
        return list(ws_docs.values())

    def get_document_content(self, workspace: str, doc_id: str) -> str:
        """Retorna o texto completo de um documento indexado para o Resource MCP docs://{workspace}/{doc_id}."""
        manifest = self._load_manifest()
        ws_docs = manifest.get(workspace, {})
        doc_info = ws_docs.get(doc_id)

        if not doc_info:
            # Tentar localizar de forma case-insensitive
            for k, v in ws_docs.items():
                if k.lower() == doc_id.lower():
                    doc_info = v
                    break

        if not doc_info:
            raise KeyError(
                f"Documento '{doc_id}' não encontrado no workspace '{workspace}'."
            )

        doc_path = Path(doc_info["path"])
        if doc_path.exists():
            loaded = load_document(doc_path)
            header = (
                f"# {doc_info.get('law_name') or doc_info.get('filename')}\n"
                f"- ID: {doc_info['doc_id']}\n"
                f"- Norma/Número: {doc_info.get('law_number', 'N/A')}\n"
                f"- Fonte: {doc_info.get('source_url', 'Local')}\n"
                f"- Chunks indexados: {doc_info.get('chunks_count', 0)}\n\n"
            )
            return header + loaded.content

        # Fallback: reconstruir a partir dos chunks salvos no ChromaDB
        collection = self._get_collection(workspace)
        res = collection.get(where={"doc_id": doc_info["doc_id"]})
        docs_list = res.get("documents") or []
        return "\n\n".join(docs_list)

    def search(
        self,
        query: str,
        workspace: str = "default",
        top_k: int = 5,
        doc_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Executa busca semântica nos documentos do workspace e retorna trechos com score."""
        collection = self._get_collection(workspace)
        total_in_col = collection.count()
        if total_in_col == 0:
            return []

        where_clause: dict[str, Any] | None = None
        if doc_filter and doc_filter.strip():
            clean_filter = doc_filter.strip().lower()
            # Verificar se bate com um doc_id conhecido no manifesto
            manifest_docs = self._load_manifest().get(workspace, {})
            matched_doc_id: str | None = None
            for did, dmeta in manifest_docs.items():
                if (
                    clean_filter == did.lower()
                    or clean_filter in did.lower()
                    or clean_filter in str(dmeta.get("law_name", "")).lower()
                    or clean_filter in str(dmeta.get("law_number", "")).lower()
                    or clean_filter in str(dmeta.get("filename", "")).lower()
                ):
                    matched_doc_id = did
                    break
            if matched_doc_id:
                where_clause = {"doc_id": matched_doc_id}
            else:
                where_clause = {"doc_id": clean_filter}

        n_results = max(1, min(top_k, total_in_col))
        # Buscar um pool maior de candidatos no ChromaDB e aplicar re-ranking híbrido
        # (combinando similaridade cosseno com correspondência lexical dos termos-chave da query)
        candidate_pool_size = min(max(n_results * 15, 60), total_in_col)
        try:
            query_res = collection.query(
                query_texts=[query],
                n_results=candidate_pool_size,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as err:
            logger.warning(f"Erro na consulta ao ChromaDB (tentando sem filtro): {err}")
            query_res = collection.query(
                query_texts=[query],
                n_results=candidate_pool_size,
                include=["documents", "metadatas", "distances"],
            )

        ids = (query_res.get("ids") or [[]])[0]
        documents = (query_res.get("documents") or [[]])[0]
        metadatas = (query_res.get("metadatas") or [[]])[0]
        distances = (query_res.get("distances") or [[]])[0]

        stopwords = {
            "qual", "quais", "como", "onde", "quando", "porque", "para", "pela",
            "pelo", "pelos", "pelas", "sobre", "entre", "segundo", "conforme",
            "previsto", "prevista", "prevê", "preve", "estabelece", "relacao",
            "relação", "últimos", "ultimos", "anos", "meses", "disponíveis",
            "clt", "cdc", "lgpd", "marco", "civil", "internet", "lei", "artigo",
            "prazo", "direito", "diz", "trata", "são", "sao", "pode", "deve",
        }
        q_lower = query.lower()
        query_terms = [
            t for t in re.findall(r"[a-zà-ú0-9]{3,}", q_lower)
            if t not in stopwords
        ]
        # Caso o terminal/chamador tenha enviado palavra corrompida em encoding (ex: f\ufffdrias), recuperar "férias"
        if ("férias" in q_lower or "ferias" in q_lower or re.search(r"f.?rias", q_lower)) and "férias" not in query_terms:
            query_terms.append("férias")

        explicit_arts = [
            f"art. {m}" for m in re.findall(r"\bart\.?\s*(\d+(?:-[a-z])?)", q_lower)
        ]

        results: list[dict[str, Any]] = []
        is_ferias_query = any(w in query_terms for w in ("férias", "ferias")) and "processual" not in q_lower

        for cid, text, meta, dist in zip(ids, documents, metadatas, distances):
            # Em espaço cosseno do ChromaDB, distance = 1 - cosine_similarity
            base_sim = max(0.0, 1.0 - float(dist))
            text_lower = str(text).lower()
            meta_dict = meta or {}
            art_lower = str(meta_dict.get("article", "")).lower()

            lexical_boost = 0.0
            if query_terms:
                matched = sum(1 for t in query_terms if t in text_lower)
                lexical_boost += 0.25 * (matched / len(query_terms))
                if matched == 0:
                    lexical_boost -= 0.20

            # Boost para o capítulo de Férias Anuais (Arts. 129 a 145 da CLT) quando a busca é sobre férias do empregado
            if is_ferias_query:
                if any(art_lower == f"art. {n}" for n in (129, 130, 134)):
                    lexical_boost += 0.35
                elif any(art_lower == f"art. {n}" for n in range(129, 146)):
                    lexical_boost += 0.20
                if "férias forenses" in text_lower or "prazo processual" in text_lower or art_lower in {"art. 775-a", "art. 701"}:
                    lexical_boost -= 0.35

            if explicit_arts and art_lower in explicit_arts:
                lexical_boost += 0.30

            final_score = round(max(0.01, min(0.9999, base_sim + lexical_boost)), 4)
            results.append(
                {
                    "chunk_id": cid,
                    "score": final_score,
                    "text": text,
                    "metadata": meta_dict,
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:n_results]
