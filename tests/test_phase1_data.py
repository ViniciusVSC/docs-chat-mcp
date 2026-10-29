"""Testes automatizados para a Fase 1 (Download e organização dos dados de demonstração)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.download_samples import clean_legislation_html


def test_clean_legislation_html_removes_strike_and_extracts_articles() -> None:
    """Garante que trechos revogados (<strike>) são removidos e artigos/metadados extraídos."""
    sample_html = """
    <html>
    <head><title>Lei Teste</title></head>
    <body>
      <p>TÍTULO I</p>
      <p>DAS DISPOSIÇÕES PRELIMINARES</p>
      <p>CAPÍTULO I</p>
      <p>DO OBJETO</p>
      <p><strike>Art. 1º Texto antigo revogado.</strike></p>
      <p>Art. 1º Esta Lei dispõe sobre o tratamento de dados pessoais.</p>
      <p>Parágrafo único. As normas gerais contidas nesta Lei são de interesse nacional.</p>
      <p>Art. 2º A disciplina da proteção de dados pessoais tem como fundamentos:</p>
      <p>I - o respeito à privacidade;</p>
    </body>
    </html>
    """.encode("cp1252")

    clean_text, articles = clean_legislation_html(
        html_bytes=sample_html,
        law_id="lei_teste",
        law_name="Lei de Teste",
        law_number="Lei nº 9.999/2024",
        source_url="https://www.planalto.gov.br/lei_teste.htm",
        download_date="2026-10-29",
    )

    assert "Texto antigo revogado" not in clean_text
    assert "Esta Lei dispõe sobre o tratamento de dados pessoais." in clean_text
    assert len(articles) == 2
    assert articles[0]["article"] == "Art. 1"
    assert articles[0]["law_name"] == "Lei de Teste"
    assert articles[0]["law_number"] == "Lei nº 9.999/2024"
    assert articles[0]["source_url"] == "https://www.planalto.gov.br/lei_teste.htm"
    assert "Parágrafo único" in articles[0]["text"]
    assert articles[1]["article"] == "Art. 2"
    assert "I - o respeito à privacidade;" in articles[1]["text"]


def test_downloaded_legislation_files_exist_and_have_content() -> None:
    """Verifica se os 4 diplomas legais foram baixados e processados em data/docs/."""
    docs_dir = Path("data/docs")
    assert docs_dir.exists(), "Diretório data/docs não encontrado."

    catalog_path = docs_dir / "metadata_catalog.json"
    assert catalog_path.exists(), "Catálogo metadata_catalog.json não encontrado."

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    expected_ids = {"clt", "cdc", "lgpd", "marco_civil_internet"}
    found_ids = {item["id"] for item in catalog}
    assert expected_ids == found_ids

    for item in catalog:
        txt_file = docs_dir / item["file"]
        json_file = docs_dir / item["articles_file"]
        assert txt_file.exists(), f"Arquivo {txt_file} ausente."
        assert json_file.exists(), f"Arquivo {json_file} ausente."

        text_content = txt_file.read_text(encoding="utf-8")
        assert len(text_content) > 10_000, f"Texto de {item['id']} parece incompleto."
        assert "url_origem:" in text_content

        articles = json.loads(json_file.read_text(encoding="utf-8"))
        assert len(articles) >= 25, f"Poucos artigos extraídos para {item['id']}: {len(articles)}"
        first_art = articles[0]
        for key in ("law_name", "law_number", "article", "source_url", "download_date", "text"):
            assert key in first_art and first_art[key]


def test_downloaded_bcb_and_ibge_tables_exist_and_are_valid() -> None:
    """Verifica se as tabelas CSV do BCB e IBGE foram geradas com colunas normalizadas e catálogo."""
    tables_dir = Path("data/tables")
    assert tables_dir.exists(), "Diretório data/tables não encontrado."

    catalog_path = tables_dir / "catalog.json"
    assert catalog_path.exists(), "Catálogo data/tables/catalog.json não encontrado."

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    table_names = {entry["table_name"] for entry in catalog}
    expected_tables = {
        "bcb_selic_meta",
        "bcb_ipca_mensal",
        "bcb_dolar_ptax_venda",
        "ibge_ufs",
        "ibge_municipios",
        "ibge_populacao_municipios",
        "ibge_pib_municipios",
    }
    assert expected_tables.issubset(table_names)

    # Verificar colunas e tipos das séries do BCB
    for bcb_table in ("bcb_selic_meta", "bcb_ipca_mensal", "bcb_dolar_ptax_venda"):
        csv_path = tables_dir / f"{bcb_table}.csv"
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["data", "valor"]
            rows = list(reader)
            assert len(rows) > 50, f"Série {bcb_table} tem poucas linhas ({len(rows)})."
            # Validar formato de data ISO e float no primeiro e último registro
            assert len(rows[0]["data"]) == 10 and rows[0]["data"][4] == "-"
            float(rows[0]["valor"])
            float(rows[-1]["valor"])

    # Verificar UFs (27 unidades federativas)
    with (tables_dir / "ibge_ufs.csv").open("r", encoding="utf-8") as f:
        uf_rows = list(csv.DictReader(f))
        assert len(uf_rows) == 27
        siglas = {r["uf_sigla"] for r in uf_rows}
        assert {"PR", "SP", "RJ", "DF"}.issubset(siglas)

    # Verificar População e PIB municipal (> 5500 municípios)
    for ibge_mun_table, val_col in (
        ("ibge_populacao_municipios", "populacao_estimada"),
        ("ibge_pib_municipios", "pib_mil_reais"),
    ):
        csv_path = tables_dir / f"{ibge_mun_table}.csv"
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            for col in reader.fieldnames:
                assert col == col.lower() and " " not in col, f"Coluna fora de snake_case: {col}"
            rows = list(reader)
            assert len(rows) > 5500, f"Tabela {ibge_mun_table} incompleta: {len(rows)} linhas"
            float(rows[0][val_col])
