"""Executor da suíte de avaliação (`python -m eval.run`).

Reporta:
1. Recall@k do retrieval nas perguntas de legislação (documentos).
2. Taxa de acerto das respostas e citações de fontes nas 20 perguntas (10 leis + 10 tabelas).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from src.agent.graph import ask_agent_async, detect_doc_filter_from_question
from src.config import load_config
from src.ingestion.indexer import DocumentVectorStore
from src.mcp_server.server import ingest

console = Console()


async def evaluate_question(
    item: dict[str, Any],
    store: DocumentVectorStore,
    top_k: int = 5,
    workspace: str = "default",
) -> dict[str, Any]:
    """Avalia uma questão de `eval/questions.yaml` medindo retrieval recall@k e acerto da resposta."""
    qid = item["id"]
    category = item["category"]
    question = item["question"]
    expected_keywords: list[str] = item.get("expected_keywords") or []

    # 1. Avaliação de Retrieval Recall@k para perguntas de texto
    retrieval_hit = False
    retrieved_articles: list[str] = []

    if category == "text":
        expected_doc_id = item.get("expected_doc_id")
        expected_articles = [a.lower() for a in (item.get("expected_articles") or [])]
        doc_filter = detect_doc_filter_from_question(question) or expected_doc_id
        search_res = store.search(
            query=question,
            workspace=workspace,
            top_k=top_k,
            doc_filter=doc_filter,
        )
        for chunk in search_res:
            meta = chunk.get("metadata") or {}
            art = str(meta.get("article", ""))
            retrieved_articles.append(art)
            if meta.get("doc_id") == expected_doc_id and art.lower() in expected_articles:
                retrieval_hit = True
    else:
        retrieval_hit = True

    # 2. Avaliação ponta a ponta do Agente LangGraph
    agent_state = await ask_agent_async(question=question, workspace=workspace)
    answer = agent_state.get("answer", "")
    sources = agent_state.get("sources") or []
    ans_lower = answer.lower()

    # Verificar presença de pelo menos uma palavra-chave/valor esperado e da fonte esperada
    keyword_matches = [kw for kw in expected_keywords if kw.lower() in ans_lower]
    has_expected_content = len(keyword_matches) >= max(1, len(expected_keywords) // 2)

    has_expected_source = False
    if category == "text":
        exp_doc = str(item.get("expected_doc_id", "")).lower()
        exp_arts = [a.lower() for a in (item.get("expected_articles") or [])]
        for s in sources:
            if s.get("type") == "document":
                art_s = str(s.get("article", "")).lower()
                if art_s in exp_arts or exp_doc in str(s.get("law_name", "")).lower():
                    has_expected_source = True
                    break
    else:
        exp_tbl = str(item.get("expected_table", "")).lower()
        for s in sources:
            if s.get("type") == "table" and exp_tbl in str(s.get("table_name", "")).lower():
                has_expected_source = True
                break

    answer_correct = bool(has_expected_content and has_expected_source)

    return {
        "id": qid,
        "category": category,
        "question": question,
        "route_predicted": agent_state.get("route"),
        "retrieval_hit": retrieval_hit,
        "retrieved_articles": retrieved_articles[:top_k],
        "answer_correct": answer_correct,
        "matched_keywords": keyword_matches,
        "sources_count": len(sources),
    }


async def run_evaluation_async(
    questions_file: Path = Path("eval/questions.yaml"),
    top_k: int = 5,
    workspace: str = "default",
) -> dict[str, Any]:
    """Executa a suíte completa de avaliação e imprime o relatório formatado."""
    cfg = load_config()
    docs_dir = Path(cfg.storage.docs_dir)
    if docs_dir.exists():
        ingest(path=str(docs_dir), workspace=workspace)

    store = DocumentVectorStore(config=cfg)
    raw_yaml = yaml.safe_load(questions_file.read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = raw_yaml.get("questions") or []

    results: list[dict[str, Any]] = []
    for item in questions:
        res = await evaluate_question(item=item, store=store, top_k=top_k, workspace=workspace)
        results.append(res)

    text_results = [r for r in results if r["category"] == "text"]
    table_results = [r for r in results if r["category"] in ("table", "mixed")]

    recall_at_k = (
        sum(1 for r in text_results if r["retrieval_hit"]) / len(text_results)
        if text_results
        else 0.0
    )
    text_accuracy = (
        sum(1 for r in text_results if r["answer_correct"]) / len(text_results)
        if text_results
        else 0.0
    )
    table_accuracy = (
        sum(1 for r in table_results if r["answer_correct"]) / len(table_results)
        if table_results
        else 0.0
    )
    overall_accuracy = (
        sum(1 for r in results if r["answer_correct"]) / len(results)
        if results
        else 0.0
    )

    table = Table(title=f"Relatório de Avaliação docs-chat-mcp (Top-{top_k})")
    table.add_column("ID", style="cyan")
    table.add_column("Categoria", style="magenta")
    table.add_column("Rota", style="blue")
    table.add_column(f"Recall@{top_k}", justify="center")
    table.add_column("Resposta + Fonte", justify="center")
    table.add_column("Artigos / Termos Validados", style="dim")

    for r in results:
        rec_str = "[green]OK[/green]" if r["retrieval_hit"] else "[red]FALHA[/red]"
        ans_str = "[green]OK[/green]" if r["answer_correct"] else "[red]FALHA[/red]"
        details = (
            ", ".join(r["retrieved_articles"][:3])
            if r["category"] == "text"
            else ", ".join(r["matched_keywords"][:3])
        )
        table.add_row(
            r["id"],
            r["category"],
            str(r["route_predicted"]),
            rec_str,
            ans_str,
            details,
        )

    console.print(table)
    console.print(
        f"\n[bold green]Resumo das Métricas:[/bold green]\n"
        f"- Total de Perguntas Avaliadas: [bold]{len(results)}[/bold] ({len(text_results)} leis, {len(table_results)} tabelas/mistas)\n"
        f"- Retrieval Recall@{top_k} (Legislação): [bold cyan]{recall_at_k * 100:.1f}%[/bold cyan]\n"
        f"- Taxa de Acerto (Legislação): [bold cyan]{text_accuracy * 100:.1f}%[/bold cyan]\n"
        f"- Taxa de Acerto (Tabelas/SQL): [bold cyan]{table_accuracy * 100:.1f}%[/bold cyan]\n"
        f"- Taxa de Acerto Geral: [bold green]{overall_accuracy * 100:.1f}%[/bold green]\n"
    )

    summary = {
        "total_questions": len(results),
        "top_k": top_k,
        "recall_at_k": round(recall_at_k, 4),
        "text_accuracy": round(text_accuracy, 4),
        "table_accuracy": round(table_accuracy, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "details": results,
    }
    report_path = Path("eval/last_report.json")
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa a avaliação de Retrieval e Respostas do docs-chat-mcp.")
    parser.add_argument("--top-k", type=int, default=5, help="Valor de K para o Recall@K (padrão: 5).")
    parser.add_argument("--workspace", type=str, default="default", help="Workspace avaliado (padrão: default).")
    args = parser.parse_args()
    asyncio.run(run_evaluation_async(top_k=args.top_k, workspace=args.workspace))


if __name__ == "__main__":
    main()
