"""Testes automatizados para a Fase 5 (Avaliação eval/questions.yaml, Streamlit helpers e Dockerfiles)."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.app import _sanitize_ws


def test_eval_questions_yaml_has_at_least_10_law_and_10_table_questions() -> None:
    """Garante que eval/questions.yaml possui >= 10 perguntas de leis e >= 10 de tabelas com gabarito."""
    q_path = Path("eval/questions.yaml")
    assert q_path.exists(), "Arquivo eval/questions.yaml não encontrado."

    data = yaml.safe_load(q_path.read_text(encoding="utf-8"))
    questions = data.get("questions") or []
    assert len(questions) >= 20

    law_questions = [q for q in questions if q.get("category") == "text"]
    table_questions = [q for q in questions if q.get("category") in ("table", "mixed")]

    assert len(law_questions) >= 10, f"Esperado >= 10 perguntas sobre leis, encontrado {len(law_questions)}"
    assert len(table_questions) >= 10, f"Esperado >= 10 perguntas sobre tabelas, encontrado {len(table_questions)}"

    for q in law_questions:
        assert q.get("question")
        assert q.get("expected_doc_id")
        assert q.get("expected_articles")
        assert q.get("expected_keywords")

    for q in table_questions:
        assert q.get("question")
        assert q.get("expected_table")
        assert q.get("expected_keywords")


def test_streamlit_and_docker_artifacts_exist() -> None:
    """Verifica existência e integridade dos arquivos da Fase 5 (Streamlit, Dockerfile, docker-compose.yml)."""
    assert Path("src/app.py").exists()
    assert Path("Dockerfile").exists()
    assert Path("docker-compose.yml").exists()
    assert _sanitize_ws("  meu workspace! ") == "meu_workspace_"
    assert _sanitize_ws("") == "default"
