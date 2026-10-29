"""Interface de linha de comando (CLI) para conversar com documentos e tabelas via agente LangGraph + MCP."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.graph import ask_agent
from src.config import load_config
from src.mcp_server.server import ingest

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI do docs-chat-mcp: converse com seus documentos e tabelas pelo terminal."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Pergunta única a ser respondida (se omitida, abre o modo chat interativo).",
    )
    parser.add_argument(
        "--workspace",
        default="default",
        help="Nome do workspace (padrão: default).",
    )
    parser.add_argument(
        "--ingest-dir",
        default=None,
        help="Caminho opcional de arquivo ou pasta para indexar antes de perguntar.",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.ingest_dir:
        console.print(f"[cyan]Indexando '{args.ingest_dir}' no workspace '{args.workspace}'...[/cyan]")
        ingest(path=args.ingest_dir, workspace=args.workspace)
    elif args.workspace == "default" and Path(cfg.storage.docs_dir).exists():
        ingest(path=cfg.storage.docs_dir, workspace="default")

    if args.question:
        state = ask_agent(question=args.question, workspace=args.workspace)
        console.print(
            Panel(
                Markdown(state.get("answer", "")),
                title=f"Resposta (Rota: {state.get('route')})",
                border_style="green",
            )
        )
        return

    console.print(
        Panel(
            "Bem-vindo ao [bold cyan]docs-chat-mcp[/bold cyan]!\n"
            "Digite sua pergunta sobre os documentos ou tabelas (ou `sair` para encerrar).",
            title="Modo Interativo",
            border_style="cyan",
        )
    )
    while True:
        try:
            q = console.input("\n[bold yellow]Pergunta > [/bold yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"sair", "exit", "quit"}:
            break
        state = ask_agent(question=q, workspace=args.workspace)
        console.print(
            Panel(
                Markdown(state.get("answer", "")),
                title=f"Resposta (Rota: {state.get('route')})",
                border_style="green",
            )
        )


if __name__ == "__main__":
    main()
