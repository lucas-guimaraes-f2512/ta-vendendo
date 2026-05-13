# Tá vendendo? — Copiloto de IA para Vendedores de Marketplace

## O que é este projeto

O "Tá vendendo?" é um copiloto de inteligência artificial embarcado no painel de vendedores de marketplace (Mercado Livre, Amazon Brasil). Ele responde perguntas em linguagem natural como:

- "Devo baixar o preço deste produto hoje?"
- "Por que minhas vendas caíram essa semana?"
- "Quais produtos posso combinar para aumentar o ticket médio?"

O sistema combina modelos preditivos clássicos com um LLM (Large Language Model) para transformar dados complexos em respostas diretas e acionáveis.

---

## Persona principal

**Carlos**, 38 anos, vendedor de eletrônicos no Mercado Livre há 4 anos. Não é técnico, não interpreta gráficos sozinho. Quer respostas diretas que ele possa agir imediatamente.

---

## Arquitetura em três camadas

### 1. Dados
- Ingestão via API das plataformas (Mercado Livre, Amazon)
- ETL em Python puro (pandas)
- Armazenamento em CSV/JSON local (fase de protótipo)
- Durante o desenvolvimento: **dados sintéticos** gerados por `generate_synthetic.py`

### 2. Inteligência
- **Prophet**: previsão de tendências de vendas
- **XGBoost**: score de risco de churn por produto
- **Regressão**: elasticidade de preço (impacto de variação de preço nas vendas)
- **LLM + RAG**: arquitetura de recuperação aumentada por contexto usando ChromaDB local

### 3. Apresentação
- Interface Streamlit com chat em linguagem natural
- Dashboards interativos com Plotly
- Sem framework pesado de backend — Python puro

---

## Decisões técnicas (Sprint 1)

| Decisão | Escolha | Motivo |
|---|---|---|
| LLM | Claude Sonnet (Anthropic) — a confirmar na Sprint 2 | Custo-benefício, API estável |
| RAG | ChromaDB local | Sem custo, roda offline, sem dependência de nuvem |
| Dados de dev | Sintéticos | Camada de anonimização LGPD ainda não implementada |
| Frontend | Streamlit | Rápido para protótipo, sem necessidade de JS |
| Backend | Python puro | Sem overhead de framework pesado nesta fase |

---

## Estrutura de pastas

```
ta-vendendo/
├── CLAUDE.md                  ← este arquivo
├── requirements.txt           ← dependências Python
├── data/
│   ├── synthetic/             ← dados gerados por generate_synthetic.py
│   └── processed/             ← dados após ETL/pipeline
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py   ← gerador de dados de desenvolvimento
│   │   └── pipeline.py             ← ETL e transformações
│   ├── models/
│   │   ├── sales_forecast.py       ← Prophet: previsão de tendências
│   │   ├── churn_risk.py           ← XGBoost: risco de abandono de produto
│   │   └── price_elasticity.py     ← regressão: elasticidade de preço
│   ├── ai/
│   │   ├── rag.py                  ← orquestração RAG
│   │   ├── vector_store.py         ← interface com ChromaDB
│   │   └── llm_client.py          ← cliente LLM (Anthropic/OpenAI)
│   └── app.py                 ← aplicação Streamlit principal
└── outputs/
    ├── sprint_2/              ← logs e artefatos da Sprint 2
    ├── sprint_3/              ← logs e artefatos da Sprint 3
    └── sprint_4/              ← logs e artefatos da Sprint 4
```

---

## Sprints planejadas

| Sprint | Foco |
|---|---|
| Sprint 1 | Estrutura, dados sintéticos, setup de ambiente |
| Sprint 2 | Modelos preditivos (Prophet, XGBoost, elasticidade) |
| Sprint 3 | RAG + LLM: chat em linguagem natural |
| Sprint 4 | Interface Streamlit completa + dashboards |

---

## Como rodar (sem conhecimento técnico)

### Pré-requisitos
1. Ter Python instalado (versão 3.10 ou superior)
2. Ter o terminal PowerShell aberto na pasta `ta-vendendo`

### Instalação das dependências (fazer uma vez só)
```
pip install -r requirements.txt
```

### Gerar os dados sintéticos
```
python src/data/generate_synthetic.py
```
Isso vai criar os arquivos em `data/synthetic/`.

### Rodar a aplicação
```
streamlit run src/app.py
```
O navegador vai abrir automaticamente com a interface.

### Onde ficam os resultados de cada sprint
Cada sprint salva um arquivo de log em `outputs/sprint_X/sprint_X_log.md` com o que foi feito, métricas e caminhos dos arquivos gerados.

---

## Variáveis de ambiente necessárias (Sprint 3 em diante)

Crie um arquivo `.env` na raiz do projeto com:
```
ANTHROPIC_API_KEY=sua_chave_aqui
OPENAI_API_KEY=sua_chave_aqui  # opcional, caso troque o LLM
```

---

## Notas de conformidade (LGPD)

- Em produção, todos os dados de vendedores reais devem passar por anonimização antes de entrar no pipeline
- O módulo de anonimização será implementado na Sprint 5
- Durante o desenvolvimento, usar exclusivamente dados do `data/synthetic/`
