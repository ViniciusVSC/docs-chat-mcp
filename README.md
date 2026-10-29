# 📚 docs-chat-mcp

**Assistente genérico para conversar com documentos e tabelas via Model Context Protocol (MCP), LangGraph, ChromaDB e DuckDB.**

O **docs-chat-mcp** permite que qualquer pessoa — inclusive sem conhecimento técnico — faça perguntas em português sobre seus próprios documentos (PDF, HTML, Markdown, TXT) e planilhas/tabelas (CSV), recebendo respostas fundamentadas e **sempre acompanhadas das fontes citadas** (lei e artigo exato para textos; tabela e consulta SQL para dados numéricos).

O projeto já inclui um script automatizado (`scripts/download_samples.py`) que baixa dados reais e oficiais da internet (Planalto, Banco Central e IBGE) para demonstração imediata.

---

## 🏗️ Arquitetura

```mermaid
flowchart TD
    User(["Usuário (Streamlit / Terminal / Claude Desktop)"])
    subgraph Agent["Agente LangGraph (src/agent/graph.py)"]
        Router{"1. Classificar Pergunta\n(text | table | mixed)"}
        SearchDocs["2a. Buscar Trechos\n(MCP Tool: search_docs)"]
        EvalRel{"Avaliar Relevância"}
        Rewrite["Reescrever Query\n(máx. 2 tentativas)"]
        InspectSchema["2b. Inspecionar Esquema\n(MCP Tools: list_tables / describe_table)"]
        GenSQL["Gerar & Validar SQL\n(Somente SELECT)"]
        ExecSQL["Executar Consulta\n(MCP Tool: query_table)"]
        Synthesize["3. Gerar Resposta com Fontes\n(MCP Prompt: responder_com_fontes)"]
    end

    subgraph MCPServer["Servidor MCP FastMCP (src/mcp_server/server.py)"]
        Chroma[("ChromaDB Persistente\n+ SentenceTransformers Multilíngue")]
        Duck[("DuckDB In-Memory\nSomente Leitura (CSVs)")]
        Resources["MCP Resources\ndocs://{workspace}/{doc_id}"]
    end

    User --> Router
    Router -->|text ou mixed| SearchDocs
    Router -->|table| InspectSchema
    SearchDocs <--> Chroma
    SearchDocs --> EvalRel
    EvalRel -->|Irrelevante e tentativas < 2| Rewrite
    Rewrite --> SearchDocs
    EvalRel -->|Relevante (rota text)| Synthesize
    EvalRel -->|Rota mixed| InspectSchema
    InspectSchema <--> Duck
    InspectSchema --> GenSQL --> ExecSQL
    ExecSQL <--> Duck
    ExecSQL --> Synthesize
    Synthesize --> User
```

### Componentes Principais
1. **Servidor MCP (`src/mcp_server/server.py`)**: construído com o SDK oficial `mcp` (`FastMCP`), expondo:
   - **Tools**:
     - `ingest(path, workspace)`: lê arquivos (`.pdf`, `.html`, `.md`, `.txt`), divide em chunks (por artigo para leis; com overlap para documentos gerais) e indexa no ChromaDB. Reindexa apenas arquivos alterados (controle por hash `SHA-256`).
     - `list_documents(workspace)`: lista todos os documentos indexados no workspace.
     - `search_docs(query, workspace, top_k, doc_filter)`: busca semântica híbrida retornando trechos, metadados e score.
     - `list_tables(workspace)` e `describe_table(name, workspace)`: inspecionam o catálogo e o esquema de colunas/tipos das tabelas CSV no DuckDB.
     - `query_table(sql, workspace)`: executa consultas SQL **somente leitura** (`SELECT` / `WITH`) no DuckDB com `timeout`, limite de linhas (`max_rows`) e bloqueio de comandos destrutivos (`DROP`, `UPDATE`, `DELETE`, `INSERT`, `ALTER`, `COPY`, `ATTACH`, múltiplos statements, etc.).
   - **Resources**: `docs://{workspace}/{doc_id}` para leitura integral de cada documento indexado.
   - **Prompt MCP**: `responder_com_fontes`, instruindo o modelo a citar obrigatoriamente as fontes e nunca inventar fatos.
2. **Agente em LangGraph (`src/agent/graph.py`)**: consome o servidor MCP via `langchain-mcp-adapters`, classifica a pergunta, avalia a relevância dos trechos recuperados (reescrevendo a busca até 2 vezes se necessário), gera/valida consultas SQL e combina os resultados.
3. **Ingestão e Chunking (`src/ingestion/`)**: suporte nativo a HTML, PDF (`PyMuPDF`), Markdown e TXT, com detector automático de textos legislativos para chunking por artigo preservando a hierarquia (`Título > Capítulo > Seção > Artigo`).

---

## 🚀 Como Instalar

### Pré-requisitos
- **Python 3.11+**
- Gerenciador [**uv**](https://docs.astral.sh/uv/) (recomendado) ou `pip + venv`

### Opção 1: Usando `uv` (Recomendado)
```bash
# 1. Instalar dependências e criar o ambiente virtual (.venv)
uv sync --extra dev

# 2. Copiar o arquivo de variáveis de ambiente
cp .env.example .env
```

### Opção 2: Usando `pip + venv`
```bash
python -m venv .venv
# No Windows (PowerShell):
.venv\Scripts\Activate.ps1
# No Linux/macOS:
# source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env
```

### Configuração do LLM (`config.yaml` e `.env`)
No arquivo `config.yaml` você pode escolher entre:
- **Claude (Anthropic)** — padrão (`provider: "anthropic"`): defina sua chave `ANTHROPIC_API_KEY` no arquivo `.env`.
- **Ollama Local** (`provider: "ollama"`): aponte para `http://localhost:11434` e escolha o modelo (ex.: `llama3.1:8b`).
- *Nota*: Caso nenhuma chave de API ou instância Ollama esteja ativa, o agente conta com um modo analítico determinístico local que permite executar todos os testes e consultas de demonstração offline.

---

## 📥 Como Baixar os Dados de Demonstração

Execute o comando único abaixo para baixar e organizar todas as leis e tabelas reais da internet:

```bash
uv run python scripts/download_samples.py
```

O script é **idempotente** (se executado novamente, pula os arquivos já baixados; use `--force` para forçar novo download), identifica-se com `User-Agent`, respeita pausas entre requisições e realiza retentativas automáticas. Os dados ficam em `data/` (ignorado no Git):

### Fontes Oficiais Baixadas e Registro de Verificação
1. **A) Legislação Brasileira (`data/docs/`) — Portal da Legislação (`planalto.gov.br`)**:
   - **CLT** (Decreto-Lei nº 5.452/1943): `https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452compilado.htm` (1.021 artigos extraídos).
   - **Código de Defesa do Consumidor — CDC** (Lei nº 8.078/1990): `https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm` (133 artigos extraídos).
   - **LGPD** (Lei nº 13.709/2018): `https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm` (83 artigos extraídos).
   - **Marco Civil da Internet** (Lei nº 12.965/2014): `https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2014/lei/l12965.htm` (34 artigos extraídos).
     - *Registro de verificação de URL*: O portal do Planalto retorna HTTP 404 para o sufixo `l12965compilado.htm`, pois mantém o texto compilado e atualizado diretamente em `l12965.htm` (marcando redações revogadas com tags `<strike>`). O nosso parser remove todas as tags `<strike>`, `<s>` e `<del>`, preservando apenas o texto vigente limpo.
2. **B) Indicadores Econômicos (`data/tables/`) — API SGS do Banco Central (`api.bcb.gov.br`)**:
   - `bcb_selic_meta.csv` (Série **432** — Meta Selic definida pelo Copom, `% a.a.`, diária, baixada em janelas de 4 anos).
   - `bcb_ipca_mensal.csv` (Série **433** — IPCA variação mensal, `% a.m.`).
   - `bcb_dolar_ptax_venda.csv` (Série **10813** — Dólar americano PTAX venda, `R$/US$`, diária, baixada em janelas de 4 anos).
3. **C) Dados Demográficos e Econômicos (`data/tables/`) — APIs do IBGE (`servicodados.ibge.gov.br`)**:
   - `ibge_ufs.csv`: 27 Unidades da Federação (API de Localidades v1).
   - `ibge_municipios.csv`: 5.571 municípios brasileiros com UF e região (API de Localidades v1).
   - `ibge_populacao_municipios.csv`: População residente estimada por município (SIDRA Agregado **6579**, variável **9324**).
   - `ibge_pib_municipios.csv`: PIB a preços correntes em Mil Reais por município (SIDRA Agregado **5938**, variável **37**).
   - `catalog.json`: catálogo descritivo de todas as séries e colunas.

---

## 💻 Como Rodar

### 1. Interface Gráfica (Streamlit)
```bash
uv run streamlit run src/app.py
```
Acesse `http://localhost:8501` no navegador. Na barra lateral você pode alternar de workspace, fazer upload de novos arquivos, inspecionar esquemas de tabelas e clicar nos botões de perguntas de exemplo.

### 2. Terminal (CLI)
Você pode fazer uma pergunta direta ou abrir o modo de chat interativo no terminal:
```bash
# Pergunta direta
uv run python -m src.cli "Qual o prazo de férias previsto na CLT?"

# Modo interativo
uv run python -m src.cli
```

### 3. Docker / Docker Compose
```bash
docker compose up --build
```
O container garante o download dos dados de demonstração em `./data` e sobe a interface Streamlit em `http://localhost:8501`.

### 4. Servidor MCP no Claude Desktop
O servidor MCP (`src/mcp_server/server.py`) funciona diretamente no **Claude Desktop** via transporte `stdio`. Adicione o trecho abaixo ao seu arquivo `claude_desktop_config.json`:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "docs-chat-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "C:/Users/vinic/Desktop/projeto_rag",
        "run",
        "python",
        "-m",
        "src.mcp_server.server",
        "--transport",
        "stdio"
      ],
      "env": {
        "DATA_DIR": "C:/Users/vinic/Desktop/projeto_rag/data"
      }
    }
  }
}
```
*(Substitua `C:/Users/vinic/Desktop/projeto_rag` pelo caminho absoluto do repositório na sua máquina).*

---

## 📂 Como Adicionar Seus Próprios Documentos e Tabelas

Você pode adicionar seus próprios arquivos (`.pdf`, `.html`, `.md`, `.txt`, `.csv`) de três formas:
1. **Pela Interface Streamlit**: na barra lateral, escolha um workspace (ex.: `meu_projeto`), selecione seus arquivos no campo *"Enviar Documentos ou Tabelas"* e clique em **"Indexar arquivos enviados"**.
2. **Pelo Terminal / CLI**:
   ```bash
   uv run python -m src.cli --workspace meu_projeto --ingest-dir caminho/para/meus_arquivos "Qual o resumo do contrato?"
   ```
3. **Via Ferramenta MCP (`ingest`)**: no Claude Desktop ou qualquer cliente MCP, peça para executar a ferramenta `ingest(path="caminho/do/arquivo.pdf", workspace="default")`. Para tabelas CSV, basta colocar o arquivo `.csv` em `data/tables/` (ou `data/tables/<workspace>/`) e ele ficará imediatamente disponível em `list_tables`, `describe_table` e `query_table`.

---

## 🧪 Testes Automatizados e Avaliação (Retrieval & QA)

### Rodar os Testes Automatizados (`pytest`)
A suíte de testes valida o download e limpeza de HTML, ingestão incremental por hash SHA-256, chunking por artigo, busca semântica, bloqueio de SQL malicioso (`DROP`, `UPDATE`, `DELETE`, `INSERT`, `COPY`, múltiplos statements) e roteamento do agente LangGraph:

```bash
uv run pytest -v
```

### Rodar a Avaliação (`python -m eval.run`)
O arquivo `eval/questions.yaml` contém **20 perguntas de avaliação** (10 sobre as leis CLT, CDC, LGPD e Marco Civil da Internet + 10 sobre as tabelas do BCB e IBGE), cada uma com resposta e fonte esperadas. Para calcular o **Recall@5** do retrieval e a **Taxa de Acerto das Respostas**:

```bash
uv run python -m eval.run
```

---

## 💬 Demonstração — Exemplos de Perguntas que Funcionam

| Tipo | Pergunta de Exemplo | Fonte Citada pelo Assistente |
| :--- | :--- | :--- |
| **Texto (CLT)** | *"Qual o prazo de férias previsto na CLT?"* | `Consolidação das Leis do Trabalho (CLT) (Decreto-Lei nº 5.452/1943) — Art. 130` |
| **Texto (LGPD)** | *"O que a LGPD diz sobre consentimento?"* | `Lei Geral de Proteção de Dados Pessoais (LGPD) (Lei nº 13.709/2018) — Art. 7 / Art. 8 / Art. 5` |
| **Tabela (BCB)** | *"Qual foi o IPCA acumulado nos últimos 12 meses disponíveis?"* | Tabela `bcb_ipca_mensal` + consulta SQL com cálculo acumulado composto e soma simples |
| **Tabela (IBGE)** | *"Quais os 10 municípios mais populosos do Paraná?"* | Tabela `ibge_populacao_municipios` + consulta SQL (`WHERE uf_sigla = 'PR' ORDER BY populacao_estimada DESC LIMIT 10`) |
| **Mista (BCB + Análise)** | *"Qual a relação entre a Selic e a inflação nos últimos 3 anos?"* | Tabelas `bcb_ipca_mensal` e `bcb_selic_meta` + SQL `JOIN` anual dos últimos 3 anos e síntese da relação |

---

## 📁 Estrutura do Repositório

```text
docs-chat-mcp/
├── src/
│   ├── config.py               # Configuração tipada (config.yaml + .env)
│   ├── app.py                  # Interface web em Streamlit
│   ├── cli.py                  # Interface de linha de comando (CLI)
│   ├── ingestion/
│   │   ├── loaders.py          # Leitores para PDF, HTML, Markdown e TXT + hash SHA-256
│   │   ├── chunker.py          # Chunking por artigo (leis) e janela com overlap (textos gerais)
│   │   └── indexer.py          # ChromaDB persistente + SentenceTransformers + re-ranking híbrido
│   ├── mcp_server/
│   │   ├── server.py           # Servidor FastMCP (tools, resources docs:// e prompt responder_com_fontes)
│   │   └── sql_engine.py       # Motor DuckDB somente leitura (validação SELECT, timeout e limite)
│   └── agent/
│       ├── llm.py              # Fábrica de LLMs (Anthropic Claude / Ollama / fallback local)
│       └── graph.py            # Grafo LangGraph conectado via langchain-mcp-adapters
├── scripts/
│   └── download_samples.py     # Download idempotente das leis (Planalto) e tabelas (BCB e IBGE)
├── eval/
│   ├── questions.yaml          # 20 perguntas de avaliação (10 leis + 10 tabelas) com gabarito
│   └── run.py                  # Executor de avaliação (Recall@k e Taxa de Acerto)
├── tests/                      # Suíte de testes automatizados (pytest) para todas as fases
├── config.yaml                 # Configurações de LLM, embeddings, chunking e DuckDB
├── Dockerfile                  # Imagem Docker da aplicação
├── docker-compose.yml          # Orquestração Docker Compose
├── pyproject.toml              # Metadados e dependências do projeto
├── .env.example                # Exemplo de variáveis de ambiente
└── README.md                   # Documentação completa em português
```
