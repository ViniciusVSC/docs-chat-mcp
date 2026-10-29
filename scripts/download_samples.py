"""Script para baixar e organizar os dados de demonstração da internet.

Fontes oficiais:
A) Legislação (Planalto - versões compiladas / texto vigente):
   - CLT (Decreto-Lei nº 5.452/1943)
   - Código de Defesa do Consumidor (Lei nº 8.078/1990)
   - LGPD (Lei nº 13.709/2018)
   - Marco Civil da Internet (Lei nº 12.965/2014)

B) Indicadores Econômicos (API SGS do Banco Central):
   - Selic meta (432) - diária (baixada em janelas de datas)
   - IPCA mensal (433) - mensal
   - Dólar PTAX venda (10813) - diária (baixada em janelas de datas)

C) Dados do IBGE (APIs de Localidades e Agregados/SIDRA):
   - Lista de UFs e Municípios (Localidades v1)
   - População estimada por município (Tabela 6579, variável 9324)
   - PIB dos municípios (Tabela 5938, variável 37)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_samples")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) docs-chat-mcp/0.1.0 "
    "(Educational RAG Demo; +https://github.com/docs-chat-mcp)"
)
REQUEST_DELAY_SECONDS = 0.6
MAX_RETRIES = 3

LAWS_SPEC: list[dict[str, str]] = [
    {
        "id": "clt",
        "law_name": "Consolidação das Leis do Trabalho (CLT)",
        "law_number": "Decreto-Lei nº 5.452/1943",
        "url": "https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452compilado.htm",
    },
    {
        "id": "cdc",
        "law_name": "Código de Defesa do Consumidor (CDC)",
        "law_number": "Lei nº 8.078/1990",
        "url": "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
    },
    {
        "id": "lgpd",
        "law_name": "Lei Geral de Proteção de Dados Pessoais (LGPD)",
        "law_number": "Lei nº 13.709/2018",
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm",
    },
    {
        "id": "marco_civil_internet",
        "law_name": "Marco Civil da Internet",
        "law_number": "Lei nº 12.965/2014",
        # Nota: O Planalto retorna 404 para l12965compilado.htm, mantendo o texto
        # compilado diretamente em l12965.htm com trechos revogados em <strike>.
        "url": "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2014/lei/l12965.htm",
    },
]

BCB_SERIES_SPEC: list[dict[str, Any]] = [
    {
        "id": "bcb_selic_meta",
        "code": 432,
        "name": "Taxa de juros - Meta Selic definida pelo Copom",
        "unit": "% a.a.",
        "frequency": "diaria",
        "start_year": 2015,
        "description": "Meta para a taxa Selic definida pelo Comitê de Política Monetária (Copom) do Banco Central do Brasil (% ao ano).",
    },
    {
        "id": "bcb_ipca_mensal",
        "code": 433,
        "name": "Índice nacional de preços ao consumidor-amplo (IPCA) - Variação mensal",
        "unit": "% a.m.",
        "frequency": "mensal",
        "start_year": 2010,
        "description": "Variação percentual mensal do IPCA calculado pelo IBGE e divulgado no SGS do Banco Central.",
    },
    {
        "id": "bcb_dolar_ptax_venda",
        "code": 10813,
        "name": "Taxa de câmbio - Livre - Dólar americano (venda)",
        "unit": "R$/US$",
        "frequency": "diaria",
        "start_year": 2015,
        "description": "Cotação diária de venda do dólar americano (PTAX) em reais (R$/US$).",
    },
]


def http_get_bytes(
    url: str,
    timeout: int = 45,
    retries: int = MAX_RETRIES,
    delay: float = REQUEST_DELAY_SECONDS,
) -> bytes:
    """Realiza requisição HTTP GET com User-Agent, descompressão gzip e retentativas."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/json,*/*;q=0.8",
        "Accept-Encoding": "gzip, identity",
    }
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if attempt > 1:
                backoff = delay * (2 ** (attempt - 1))
                logger.warning(f"Retentativa {attempt}/{retries} para {url} após {backoff:.1f}s...")
                time.sleep(backoff)
            else:
                time.sleep(delay)

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                return raw
        except Exception as err:
            last_err = err
            logger.warning(f"Erro ao acessar {url} (tentativa {attempt}/{retries}): {err}")

    raise RuntimeError(f"Falha ao baixar {url} após {retries} tentativas: {last_err}") from last_err


def http_get_json(url: str, timeout: int = 45) -> Any:
    """Baixa e decodifica um JSON a partir de uma URL."""
    raw = http_get_bytes(url, timeout=timeout)
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# A) Legislação (Planalto)
# ---------------------------------------------------------------------------

def clean_legislation_html(
    html_bytes: bytes,
    law_id: str,
    law_name: str,
    law_number: str,
    source_url: str,
    download_date: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Converte HTML do Planalto em texto limpo estruturado e extrai metadados por artigo.

    Remove tags <strike>, <s> e <del> (dispositivos revogados/alterados), scripts e estilos.
    Preserva a hierarquia (TÍTULO, CAPÍTULO, SEÇÃO, Artigo) e gera cabeçalho de metadados.
    """
    try:
        html_text = html_bytes.decode("cp1252")
    except UnicodeDecodeError:
        html_text = html_bytes.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html_text, "html.parser")

    # Remover scripts, estilos e trechos revogados/riscados
    for tag in soup.find_all(["script", "style", "strike", "s", "del", "head"]):
        tag.decompose()

    raw_text = soup.get_text(separator="\n")

    # Normalizar espaços em branco linha a linha
    cleaned_lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = re.sub(r"[ \t\xa0]+", " ", raw_line).strip()
        if not line:
            continue
        # Ignorar links de navegação comuns do Planalto no topo
        if line.lower() in {"texto compilado", "vigência", "mensagem de veto", "regulamento"}:
            continue
        cleaned_lines.append(line)

    # Reagrupar linhas que pertencem ao mesmo parágrafo/artigo caso o HTML tenha quebras no meio da frase
    merged_paragraphs: list[str] = []
    structural_pattern = re.compile(
        r"^(PARTE\b|LIVRO\b|T[ÍI]TULO\b|CAP[ÍI]TULO\b|SE[ÇC][ÃA]O\b|SUBSE[ÇC][ÃA]O\b|"
        r"Art\.?\s*\d+|Artigo\s*\d+|Par[áa]grafo\s+[úu]nico|§\s*\d+|[IVXLCDM]+\s*[-–—]|"
        r"[a-z]\)\s+|DECRETO-LEI|LEI\s+N)",
        re.IGNORECASE,
    )

    for line in cleaned_lines:
        if not merged_paragraphs:
            merged_paragraphs.append(line)
            continue

        if structural_pattern.match(line):
            merged_paragraphs.append(line)
        else:
            prev = merged_paragraphs[-1]
            # Se a linha anterior era um cabeçalho estrutural curto (ex: "CAPÍTULO I"),
            # e a linha atual é o subtítulo (ex: "DA IDENTIFICAÇÃO PROFISSIONAL"),
            # mantemos em linha separada para legibilidade.
            if re.match(
                r"^(PARTE|LIVRO|T[ÍI]TULO|CAP[ÍI]TULO|SE[ÇC][ÃA]O|SUBSE[ÇC][ÃA]O)\b",
                prev,
                re.IGNORECASE,
            ) and len(prev) < 60:
                merged_paragraphs.append(line)
            else:
                merged_paragraphs[-1] = f"{prev} {line}"

    # Extrair artigos e contexto hierárquico (título, capítulo, seção)
    articles_metadata: list[dict[str, Any]] = []
    current_title = ""
    current_chapter = ""
    current_section = ""

    current_article_num: str | None = None
    current_article_lines: list[str] = []

    art_regex = re.compile(r"^Art\.?\s*(\d+(?:-[A-Za-z])?)\s*[º°.]?", re.IGNORECASE)

    def flush_article() -> None:
        nonlocal current_article_num, current_article_lines
        if current_article_num and current_article_lines:
            articles_metadata.append(
                {
                    "law_id": law_id,
                    "law_name": law_name,
                    "law_number": law_number,
                    "article": f"Art. {current_article_num}",
                    "title": current_title,
                    "chapter": current_chapter,
                    "section": current_section,
                    "source_url": source_url,
                    "download_date": download_date,
                    "text": "\n".join(current_article_lines),
                }
            )
        current_article_num = None
        current_article_lines = []

    formatted_body_lines: list[str] = []

    for para in merged_paragraphs:
        if re.match(r"^T[ÍI]TULO\b", para, re.IGNORECASE):
            flush_article()
            current_title = para
            current_chapter = ""
            current_section = ""
            formatted_body_lines.append(f"\n## {para}")
            continue
        if re.match(r"^CAP[ÍI]TULO\b", para, re.IGNORECASE):
            flush_article()
            current_chapter = para
            current_section = ""
            formatted_body_lines.append(f"\n### {para}")
            continue
        if re.match(r"^(SE[ÇC][ÃA]O|SUBSE[ÇC][ÃA]O)\b", para, re.IGNORECASE):
            flush_article()
            current_section = para
            formatted_body_lines.append(f"\n#### {para}")
            continue

        art_match = art_regex.match(para)
        if art_match:
            flush_article()
            current_article_num = art_match.group(1)
            current_article_lines = [para]
            formatted_body_lines.append(f"\n{para}")
        else:
            if current_article_num is not None:
                current_article_lines.append(para)
            formatted_body_lines.append(para)

    flush_article()

    header = (
        f"---\n"
        f"law_id: {law_id}\n"
        f"lei: {law_name}\n"
        f"numero: {law_number}\n"
        f"url_origem: {source_url}\n"
        f"data_download: {download_date}\n"
        f"total_artigos_extraidos: {len(articles_metadata)}\n"
        f"---\n\n"
        f"# {law_name} ({law_number})\n"
    )

    full_clean_text = header + "\n".join(formatted_body_lines).strip() + "\n"
    return full_clean_text, articles_metadata


def download_legislation(docs_dir: Path, force: bool = False) -> list[dict[str, Any]]:
    """Baixa as 4 leis do Planalto, salva texto limpo (.txt) e metadados estruturados (.json)."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = docs_dir / "metadata_catalog.json"
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    catalog: list[dict[str, Any]] = []

    for idx, spec in enumerate(LAWS_SPEC, start=1):
        law_id = spec["id"]
        txt_path = docs_dir / f"{law_id}.txt"
        articles_json_path = docs_dir / f"{law_id}_articles.json"

        if not force and txt_path.exists() and articles_json_path.exists() and txt_path.stat().st_size > 1000:
            logger.info(
                f"[{idx}/{len(LAWS_SPEC)}] Legislação '{spec['law_name']}' já existe ({txt_path.name}). Pulando download."
            )
            existing_articles = json.loads(articles_json_path.read_text(encoding="utf-8"))
            catalog.append(
                {
                    "id": law_id,
                    "law_name": spec["law_name"],
                    "law_number": spec["law_number"],
                    "source_url": spec["url"],
                    "file": txt_path.name,
                    "articles_file": articles_json_path.name,
                    "articles_count": len(existing_articles),
                    "download_date": existing_articles[0]["download_date"] if existing_articles else download_date,
                }
            )
            continue

        logger.info(f"[{idx}/{len(LAWS_SPEC)}] Baixando {spec['law_name']} de {spec['url']} ...")
        html_bytes = http_get_bytes(spec["url"], timeout=60)
        clean_text, articles_meta = clean_legislation_html(
            html_bytes=html_bytes,
            law_id=law_id,
            law_name=spec["law_name"],
            law_number=spec["law_number"],
            source_url=spec["url"],
            download_date=download_date,
        )

        txt_path.write_text(clean_text, encoding="utf-8")
        articles_json_path.write_text(
            json.dumps(articles_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            f"    -> Salvo: {txt_path.name} ({len(clean_text):,} chars, {len(articles_meta)} artigos identificados)"
        )

        catalog.append(
            {
                "id": law_id,
                "law_name": spec["law_name"],
                "law_number": spec["law_number"],
                "source_url": spec["url"],
                "file": txt_path.name,
                "articles_file": articles_json_path.name,
                "articles_count": len(articles_meta),
                "download_date": download_date,
            }
        )

    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


# ---------------------------------------------------------------------------
# B) Indicadores Econômicos (API SGS do Banco Central)
# ---------------------------------------------------------------------------

def fetch_bcb_series(code: int, start_year: int, frequency: str) -> list[dict[str, Any]]:
    """Baixa série do SGS/BCB em janelas de 4 anos para séries diárias (respeitando limite da API)."""
    current_year = date.today().year
    window_years = 4 if frequency == "diaria" else 20

    records_by_date: dict[str, float] = {}
    year = start_year

    while year <= current_year:
        end_year = min(year + window_years - 1, current_year)
        data_inicial = f"01/01/{year}"
        data_final = f"31/12/{end_year}"
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
            f"?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
        )
        logger.info(f"    -> Consultando BCB SGS série {code} ({data_inicial} a {data_final})...")
        try:
            chunk = http_get_json(url, timeout=45)
        except Exception as err:
            # Caso o ano corrente no relógio do ambiente ainda não possua dados futuros no BCB
            logger.warning(f"    -> Janela {data_inicial}-{data_final} retornou aviso: {err}")
            chunk = []

        if isinstance(chunk, list):
            for item in chunk:
                raw_date = item.get("data", "")
                raw_val = item.get("valor", "")
                if not raw_date or raw_val in ("", None):
                    continue
                try:
                    # Converter DD/MM/YYYY para YYYY-MM-DD (padrão ISO para SQL/DuckDB)
                    dt_iso = datetime.strptime(raw_date, "%d/%m/%Y").strftime("%Y-%m-%d")
                    val_float = float(str(raw_val).replace(",", "."))
                    records_by_date[dt_iso] = val_float
                except ValueError:
                    continue

        year = end_year + 1

    sorted_dates = sorted(records_by_date.keys())
    return [{"data": d, "valor": records_by_date[d]} for d in sorted_dates]


def download_bcb_indicators(tables_dir: Path, force: bool = False) -> list[dict[str, Any]]:
    """Baixa as séries do Banco Central (Selic 432, IPCA 433, Dólar PTAX 10813) e gera catálogo."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    catalog_entries: list[dict[str, Any]] = []

    for idx, spec in enumerate(BCB_SERIES_SPEC, start=1):
        csv_path = tables_dir / f"{spec['id']}.csv"
        if not force and csv_path.exists() and csv_path.stat().st_size > 100:
            with csv_path.open("r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1
            logger.info(
                f"[{idx}/{len(BCB_SERIES_SPEC)}] Série BCB '{spec['id']}' já existe ({row_count} linhas). Pulando."
            )
        else:
            logger.info(f"[{idx}/{len(BCB_SERIES_SPEC)}] Baixando série BCB {spec['code']} ({spec['name']})...")
            rows = fetch_bcb_series(
                code=spec["code"],
                start_year=spec["start_year"],
                frequency=spec["frequency"],
            )
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["data", "valor"])
                writer.writeheader()
                writer.writerows(rows)
            row_count = len(rows)
            logger.info(f"    -> Salvo: {csv_path.name} ({row_count:,} linhas)")

        catalog_entries.append(
            {
                "table_name": spec["id"],
                "file": csv_path.name,
                "source": "Banco Central do Brasil - API SGS",
                "sgs_code": spec["code"],
                "name": spec["name"],
                "unit": spec["unit"],
                "frequency": spec["frequency"],
                "description": spec["description"],
                "columns": {
                    "data": "Data de referência no formato ISO (YYYY-MM-DD)",
                    "valor": f"Valor numérico da série ({spec['unit']})",
                },
                "row_count": row_count,
                "download_date": download_date,
            }
        )

    return catalog_entries


# ---------------------------------------------------------------------------
# C) Dados do IBGE (Localidades e SIDRA Agregados 6579 e 5938)
# ---------------------------------------------------------------------------

def download_ibge_tables(tables_dir: Path, force: bool = False) -> list[dict[str, Any]]:
    """Baixa UFs, municípios, população estimada (6579) e PIB municipal (5938) do IBGE."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    catalog_entries: list[dict[str, Any]] = []

    # 1. Lista de UFs
    ufs_csv = tables_dir / "ibge_ufs.csv"
    ufs_map: dict[int, dict[str, Any]] = {}

    logger.info("[1/4] Obtendo lista de UFs da API de Localidades do IBGE...")
    ufs_raw = http_get_json("https://servicodados.ibge.gov.br/api/v1/localidades/estados")
    uf_rows: list[dict[str, Any]] = []
    for u in sorted(ufs_raw, key=lambda x: int(x["id"])):
        uf_id = int(u["id"])
        regiao = u.get("regiao") or {}
        row = {
            "uf_id": uf_id,
            "uf_sigla": str(u.get("sigla", "")).strip(),
            "uf_nome": str(u.get("nome", "")).strip(),
            "regiao_id": int(regiao.get("id", 0)),
            "regiao_sigla": str(regiao.get("sigla", "")).strip(),
            "regiao_nome": str(regiao.get("nome", "")).strip(),
        }
        ufs_map[uf_id] = row
        uf_rows.append(row)

    with ufs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uf_id", "uf_sigla", "uf_nome", "regiao_id", "regiao_sigla", "regiao_nome"],
        )
        writer.writeheader()
        writer.writerows(uf_rows)
    logger.info(f"    -> Salvo: {ufs_csv.name} ({len(uf_rows)} UFs)")

    catalog_entries.append(
        {
            "table_name": "ibge_ufs",
            "file": ufs_csv.name,
            "source": "IBGE - API de Localidades v1",
            "description": "Lista oficial de Unidades da Federação (estados e Distrito Federal) e suas regiões.",
            "columns": {
                "uf_id": "Código IBGE da UF (inteiro)",
                "uf_sigla": "Sigla da UF (ex: PR, SP, RJ)",
                "uf_nome": "Nome da Unidade da Federação",
                "regiao_id": "Código IBGE da macrorregião",
                "regiao_sigla": "Sigla da macrorregião (N, NE, SE, S, CO)",
                "regiao_nome": "Nome da macrorregião",
            },
            "row_count": len(uf_rows),
            "download_date": download_date,
        }
    )

    # 2. Lista de Municípios
    mun_csv = tables_dir / "ibge_municipios.csv"
    mun_lookup: dict[int, dict[str, Any]] = {}

    logger.info("[2/4] Obtendo lista de Municípios da API de Localidades do IBGE...")
    muns_raw = http_get_json("https://servicodados.ibge.gov.br/api/v1/localidades/municipios")
    mun_rows: list[dict[str, Any]] = []
    for m in sorted(muns_raw, key=lambda x: int(x["id"])):
        mun_id = int(m["id"])
        # Extrair UF via microrregiao.mesorregiao.UF ou regiao-imediata.regiao-intermediaria.UF
        uf_obj = (
            ((m.get("microrregiao") or {}).get("mesorregiao") or {}).get("UF")
            or ((m.get("regiao-imediata") or {}).get("regiao-intermediaria") or {}).get("UF")
            or {}
        )
        reg_obj = uf_obj.get("regiao") or {}
        row = {
            "municipio_id": mun_id,
            "municipio_nome": str(m.get("nome", "")).strip(),
            "uf_id": int(uf_obj.get("id", 0)),
            "uf_sigla": str(uf_obj.get("sigla", "")).strip(),
            "uf_nome": str(uf_obj.get("nome", "")).strip(),
            "regiao_nome": str(reg_obj.get("nome", "")).strip(),
        }
        mun_lookup[mun_id] = row
        mun_rows.append(row)

    with mun_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["municipio_id", "municipio_nome", "uf_id", "uf_sigla", "uf_nome", "regiao_nome"],
        )
        writer.writeheader()
        writer.writerows(mun_rows)
    logger.info(f"    -> Salvo: {mun_csv.name} ({len(mun_rows):,} municípios)")

    catalog_entries.append(
        {
            "table_name": "ibge_municipios",
            "file": mun_csv.name,
            "source": "IBGE - API de Localidades v1",
            "description": "Lista oficial de municípios brasileiros com código IBGE, UF e macrorregião.",
            "columns": {
                "municipio_id": "Código IBGE de 7 dígitos do município (inteiro)",
                "municipio_nome": "Nome do município",
                "uf_id": "Código IBGE da UF",
                "uf_sigla": "Sigla da UF (ex: PR, SP)",
                "uf_nome": "Nome da UF (ex: Paraná, São Paulo)",
                "regiao_nome": "Nome da macrorregião",
            },
            "row_count": len(mun_rows),
            "download_date": download_date,
        }
    )

    # 3. População estimada por município (Agregado 6579, variável 9324)
    pop_csv = tables_dir / "ibge_populacao_municipios.csv"
    if not force and pop_csv.exists() and pop_csv.stat().st_size > 10000:
        with pop_csv.open("r", encoding="utf-8") as f:
            pop_count = sum(1 for _ in f) - 1
        logger.info(f"[3/4] Tabela '{pop_csv.name}' já existe ({pop_count:,} linhas). Pulando.")
    else:
        logger.info("[3/4] Baixando População Estimada por Município (SIDRA Tabela 6579, var 9324)...")
        url_6579 = (
            "https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/-1"
            "/variaveis/9324?localidades=N6[all]"
        )
        data_6579 = http_get_json(url_6579, timeout=90)
        series_list = data_6579[0]["resultados"][0]["series"]
        pop_rows: list[dict[str, Any]] = []

        for item in series_list:
            loc = item.get("localidade", {})
            mun_id = int(loc.get("id", 0))
            serie_dict = item.get("serie", {})
            if not serie_dict:
                continue
            ano_str, val_str = next(iter(serie_dict.items()))
            if not val_str or val_str in ("-", "...", "X", ".."):
                continue
            try:
                populacao = int(float(val_str))
                ano = int(ano_str)
            except ValueError:
                continue

            mun_info = mun_lookup.get(mun_id)
            if mun_info:
                mun_nome = mun_info["municipio_nome"]
                uf_sigla = mun_info["uf_sigla"]
                uf_nome = mun_info["uf_nome"]
            else:
                raw_name = str(loc.get("nome", ""))
                if " - " in raw_name:
                    mun_nome, uf_sigla = raw_name.rsplit(" - ", 1)
                else:
                    mun_nome, uf_sigla = raw_name, ""
                uf_nome = ""

            pop_rows.append(
                {
                    "municipio_id": mun_id,
                    "municipio_nome": mun_nome,
                    "uf_sigla": uf_sigla,
                    "uf_nome": uf_nome,
                    "ano": ano,
                    "populacao_estimada": populacao,
                }
            )

        pop_rows.sort(key=lambda r: r["municipio_id"])
        with pop_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "municipio_id",
                    "municipio_nome",
                    "uf_sigla",
                    "uf_nome",
                    "ano",
                    "populacao_estimada",
                ],
            )
            writer.writeheader()
            writer.writerows(pop_rows)
        pop_count = len(pop_rows)
        logger.info(f"    -> Salvo: {pop_csv.name} ({pop_count:,} municípios)")

    catalog_entries.append(
        {
            "table_name": "ibge_populacao_municipios",
            "file": pop_csv.name,
            "source": "IBGE - SIDRA Agregado 6579 (Variável 9324)",
            "sidra_table": 6579,
            "description": "População residente estimada por município brasileiro no último período disponível.",
            "columns": {
                "municipio_id": "Código IBGE de 7 dígitos do município (inteiro)",
                "municipio_nome": "Nome do município",
                "uf_sigla": "Sigla da UF (ex: PR, SP, MG)",
                "uf_nome": "Nome da Unidade da Federação (ex: Paraná, São Paulo)",
                "ano": "Ano de referência da estimativa populacional (inteiro)",
                "populacao_estimada": "População residente estimada (número de habitantes, inteiro)",
            },
            "row_count": pop_count,
            "download_date": download_date,
        }
    )

    # 4. PIB dos Municípios (Agregado 5938, variável 37 - PIB a preços correntes em Mil Reais)
    pib_csv = tables_dir / "ibge_pib_municipios.csv"
    if not force and pib_csv.exists() and pib_csv.stat().st_size > 10000:
        with pib_csv.open("r", encoding="utf-8") as f:
            pib_count = sum(1 for _ in f) - 1
        logger.info(f"[4/4] Tabela '{pib_csv.name}' já existe ({pib_count:,} linhas). Pulando.")
    else:
        logger.info("[4/4] Baixando PIB dos Municípios (SIDRA Tabela 5938, var 37)...")
        url_5938 = (
            "https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/-1"
            "/variaveis/37?localidades=N6[all]"
        )
        data_5938 = http_get_json(url_5938, timeout=90)
        series_list = data_5938[0]["resultados"][0]["series"]
        pib_rows: list[dict[str, Any]] = []

        for item in series_list:
            loc = item.get("localidade", {})
            mun_id = int(loc.get("id", 0))
            serie_dict = item.get("serie", {})
            if not serie_dict:
                continue
            ano_str, val_str = next(iter(serie_dict.items()))
            if not val_str or val_str in ("-", "...", "X", ".."):
                continue
            try:
                pib_mil_reais = float(val_str.replace(",", "."))
                ano = int(ano_str)
            except ValueError:
                continue

            mun_info = mun_lookup.get(mun_id)
            if mun_info:
                mun_nome = mun_info["municipio_nome"]
                uf_sigla = mun_info["uf_sigla"]
                uf_nome = mun_info["uf_nome"]
            else:
                raw_name = str(loc.get("nome", ""))
                if " - " in raw_name:
                    mun_nome, uf_sigla = raw_name.rsplit(" - ", 1)
                else:
                    mun_nome, uf_sigla = raw_name, ""
                uf_nome = ""

            pib_rows.append(
                {
                    "municipio_id": mun_id,
                    "municipio_nome": mun_nome,
                    "uf_sigla": uf_sigla,
                    "uf_nome": uf_nome,
                    "ano": ano,
                    "pib_mil_reais": pib_mil_reais,
                }
            )

        pib_rows.sort(key=lambda r: r["municipio_id"])
        with pib_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "municipio_id",
                    "municipio_nome",
                    "uf_sigla",
                    "uf_nome",
                    "ano",
                    "pib_mil_reais",
                ],
            )
            writer.writeheader()
            writer.writerows(pib_rows)
        pib_count = len(pib_rows)
        logger.info(f"    -> Salvo: {pib_csv.name} ({pib_count:,} municípios)")

    catalog_entries.append(
        {
            "table_name": "ibge_pib_municipios",
            "file": pib_csv.name,
            "source": "IBGE - SIDRA Agregado 5938 (Variável 37)",
            "sidra_table": 5938,
            "description": "Produto Interno Bruto (PIB) a preços correntes por município (unidade: Mil Reais).",
            "columns": {
                "municipio_id": "Código IBGE de 7 dígitos do município (inteiro)",
                "municipio_nome": "Nome do município",
                "uf_sigla": "Sigla da UF (ex: PR, SP, MG)",
                "uf_nome": "Nome da Unidade da Federação",
                "ano": "Ano de referência do PIB (inteiro)",
                "pib_mil_reais": "PIB a preços correntes em Mil Reais (float)",
            },
            "row_count": pib_count,
            "download_date": download_date,
        }
    )

    return catalog_entries


def download_all(base_dir: Path, force: bool = False) -> dict[str, Any]:
    """Executa o download completo (Legislação, BCB e IBGE) e salva os catálogos."""
    docs_dir = base_dir / "docs"
    tables_dir = base_dir / "tables"

    logger.info("=== Iniciando download do pacote de demonstração (docs-chat-mcp) ===")
    logger.info(f"Diretório de destino: {base_dir.resolve()}")

    logger.info("\n--- Etapa A: Legislação Brasileira (Planalto) ---")
    docs_catalog = download_legislation(docs_dir=docs_dir, force=force)

    logger.info("\n--- Etapa B: Indicadores Econômicos (Banco Central - API SGS) ---")
    bcb_catalog = download_bcb_indicators(tables_dir=tables_dir, force=force)

    logger.info("\n--- Etapa C: Dados Municipais e Demográficos (IBGE - Localidades & SIDRA) ---")
    ibge_catalog = download_ibge_tables(tables_dir=tables_dir, force=force)

    tables_catalog = bcb_catalog + ibge_catalog
    tables_catalog_path = tables_dir / "catalog.json"
    tables_catalog_path.write_text(
        json.dumps(tables_catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"\nCatálogo de tabelas salvo em: {tables_catalog_path}")
    logger.info("=== Download finalizado com sucesso! ===")

    return {
        "docs": docs_catalog,
        "tables": tables_catalog,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baixa e prepara os dados reais de demonstração (Planalto, BCB e IBGE)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Diretório raiz para salvar docs/ e tables/ (padrão: data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força o re-download mesmo que os arquivos já existam.",
    )
    args = parser.parse_args()
    download_all(base_dir=args.data_dir, force=args.force)


if __name__ == "__main__":
    main()
