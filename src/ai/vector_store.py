"""
Vector store com ChromaDB local.
Persistent client em data/chroma_db/, collection "ta_vendendo_carlos".
Embedding: paraphrase-multilingual-MiniLM-L12-v2 (suporte a PT-BR).

Documentos indexados:
  - perfil      : campos do seller_profile.json
  - tendencia   : linhas do forecast_summary_corrigido.csv
  - churn       : linhas do churn_scores.csv
  - limitacao   : avisos técnicos fixos
  - resumo      : resumo executivo do Carlos
"""
import os
import json
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH   = os.path.join(BASE_DIR, "data", "chroma_db")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")
SPRINT3_DIR   = os.path.join(BASE_DIR, "outputs", "sprint_3")
SPRINT2_DIR   = os.path.join(BASE_DIR, "outputs", "sprint_2")

COLLECTION_NAME  = "ta_vendendo_carlos"
EMBEDDING_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"


# ──────────────────────────────────────────────────────────────
# Embedding function (multilingual, suporte a PT-BR)
# ──────────────────────────────────────────────────────────────

def _get_embedding_fn():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


# ──────────────────────────────────────────────────────────────
# Cliente ChromaDB
# ──────────────────────────────────────────────────────────────

def _get_client():
    import chromadb
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _get_or_reset_collection(client, embedding_fn, reset: bool = False):
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  Collection '{COLLECTION_NAME}' deletada para reindexacao.")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ──────────────────────────────────────────────────────────────
# Construtores de documentos
# ──────────────────────────────────────────────────────────────

def _docs_perfil(profile: dict) -> list[dict]:
    prods = profile.get("active_products", [])
    prod_list = ", ".join(
        f"{p['name']} (R${p['current_price']:.2f}, estoque {p['stock']})"
        for p in prods
    )
    docs = [
        {
            "id":   "perfil_identidade",
            "text": (f"O vendedor Carlos Ferreira (ID: {profile['seller_id']}) "
                     f"atua no {profile['platform']} desde {profile['start_date']}, "
                     f"na categoria {profile['category']}."),
            "meta": {"tipo": "perfil", "campo": "identidade"},
        },
        {
            "id":   "perfil_reputacao",
            "text": (f"O seller Carlos tem reputacao {profile['reputation_level']}, "
                     f"health score de {profile['health_score']}/100 e "
                     f"score de risco de churn de {profile['churn_risk_score']}."),
            "meta": {"tipo": "perfil", "campo": "reputacao"},
        },
        {
            "id":   "perfil_financeiro",
            "text": (f"O GMV medio mensal do Carlos e R$ {profile['avg_monthly_gmv']:,.2f}. "
                     f"Taxa de devolucao: {profile['return_rate_pct']}%. "
                     f"Review medio: {profile['avg_review_score']} estrelas."),
            "meta": {"tipo": "perfil", "campo": "financeiro"},
        },
        {
            "id":   "perfil_logistica",
            "text": (f"O fulfillment e {profile['fulfillment_type']} "
                     f"com prazo medio de entrega de {profile['avg_shipping_days']} dias."),
            "meta": {"tipo": "perfil", "campo": "logistica"},
        },
        {
            "id":   "perfil_produtos",
            "text": (f"O Carlos tem {len(prods)} produtos ativos: {prod_list}."),
            "meta": {"tipo": "perfil", "campo": "produtos_ativos"},
        },
    ]
    return docs


def _docs_tendencias(forecast_df: pd.DataFrame) -> list[dict]:
    docs = []
    for _, row in forecast_df.iterrows():
        pid     = row["product_id"]
        nome    = row["product_name"]
        direcao = row["trend_direction"]
        atual   = row["avg_weekly_gmv_actual"]
        prev    = row["avg_weekly_gmv_forecast"]
        pct     = row["pct_change"]

        direcao_pt = {"growing": "crescimento", "stable": "estavel",
                      "declining": "queda"}.get(direcao, direcao)

        # Texto rico em linguagem natural para melhorar o recall semantico
        text = (
            f"Tendencia de vendas do produto {nome}: as vendas estao em {direcao_pt}. "
            f"O GMV medio atual e de R${atual:,.2f} por semana. "
            f"A previsao para as proximas 12 semanas e de R${prev:,.2f} por semana, "
            f"uma variacao de {pct:+.1f}% em relacao ao periodo atual."
        )
        if direcao == "declining":
            text += f" As vendas desse produto estao caindo."
        elif direcao == "growing":
            text += f" As vendas desse produto estao crescendo."

        docs.append({
            "id":   f"tendencia_{pid}",
            "text": text,
            "meta": {"tipo": "tendencia", "product_id": pid,
                     "trend_direction": direcao},
        })
    return docs


def _docs_churn(churn_df: pd.DataFrame) -> list[dict]:
    docs = []
    for _, row in churn_df.iterrows():
        pid   = row["product_id"]
        nome  = row["product_name"]
        score = row["churn_risk_score"]
        nivel = row["risk_level"]
        sinal = row["main_signal"]

        nivel_pt = {"High": "alto", "Medium": "medio", "Low": "baixo"}.get(nivel, nivel)

        # Linguagem natural com sinonimos de "risco" para ampliar o recall
        text = (
            f"O produto {nome} corre risco de ser abandonado do catalogo. "
            f"O risco de churn e {nivel_pt} (score {score:.2f} de 1.0). "
            f"Principal sinal de alerta: {sinal}. "
        )
        if nivel == "High":
            text += "Atencao urgente recomendada para nao perder esse produto."
        elif nivel == "Medium":
            text += "Monitoramento proximo recomendado para esse produto."
        else:
            text += "Produto estavel, mas deve ser monitorado."

        docs.append({
            "id":   f"churn_{pid}",
            "text": text,
            "meta": {"tipo": "churn", "product_id": pid,
                     "risk_level": nivel, "churn_risk_score": float(score)},
        })
    return docs


def _docs_limitacoes() -> list[dict]:
    return [
        {
            "id":   "limitacao_prod001_002",
            "text": ("As previsoes de vendas do Fone BT JBL T510 (PROD001) e do "
                     "Carregador Turbo 65W (PROD002) podem ter distorcao residual de "
                     "sazonalidade da Black Friday mesmo apos o tratamento aplicado. "
                     "Para esses produtos, prefira analisar os ultimos 60 dias em vez "
                     "de confiar apenas na previsao de tendencia."),
            "meta": {"tipo": "limitacao", "produto": "PROD001,PROD002"},
        },
        {
            "id":   "limitacao_elasticidade",
            "text": ("As recomendacoes de ajuste de preco foram calculadas sobre dados "
                     "sinteticos sem correlacao causal real entre preco e conversao. "
                     "Aumentar ou reduzir preco pode nao ter o efeito esperado com dados "
                     "reais. As sugestoes sao indicativas, nao conclusivas."),
            "meta": {"tipo": "limitacao", "area": "elasticidade"},
        },
        {
            "id":   "limitacao_dados_sinteticos",
            "text": ("Todos os dados usados neste sistema sao sinteticos e simulados "
                     "para desenvolvimento. Os resultados refletem padroes artificiais. "
                     "Com dados reais da API do Mercado Livre, os modelos serao "
                     "recalibrados e os resultados podem variar significativamente."),
            "meta": {"tipo": "limitacao", "area": "dados"},
        },
    ]


def _docs_concentracao(conc_df: pd.DataFrame) -> list[dict]:
    """Um documento por produto classificado como crítico ou relevante."""
    docs = []
    for _, row in conc_df.iterrows():
        if row["classificacao"] in ["crítico", "relevante"]:
            texto = (
                f"O produto {row['product_name']} representa {row['pct_receita']:.1f}% "
                f"do faturamento total do Carlos, com GMV de R${row['gmv_total_periodo']:,.2f} "
                f"no período analisado. Classificação: {row['classificacao']}. "
            )
            if row["classificacao"] == "crítico":
                texto += "Perder esse produto teria impacto severo no faturamento."
            docs.append({
                "id":   f"concentracao_{row['product_id']}",
                "text": texto,
                "meta": {"tipo": "concentracao", "product_id": row["product_id"]},
            })
    return docs


def _docs_sazonalidade(season_df: pd.DataFrame) -> list[dict]:
    """Um documento por mês fora do padrão normal (pico ou baixo)."""
    docs = []
    for _, row in season_df.iterrows():
        if row["classificacao"] != "normal":
            texto = (
                f"{row['mes_nome']} é historicamente um mês de {row['classificacao']} "
                f"para as vendas do Carlos. O índice sazonal é {row['indice_sazonal']:.2f} "
                f"(média geral = 1.00), com GMV médio de R${row['gmv_medio_mensal']:,.2f}. "
            )
            if row["classificacao"] == "pico":
                texto += "É recomendável aumentar o estoque antes desse período."
            else:
                texto += "As vendas tendem a ser menores nesse mês — ajuste expectativas e custos."
            docs.append({
                "id":   f"sazonalidade_mes_{row['mes']:02d}",
                "text": texto,
                "meta": {"tipo": "sazonalidade", "mes": int(row["mes"])},
            })
    return docs


def _docs_receita_liquida(net_df: pd.DataFrame) -> list[dict]:
    """Um documento com resumo de receita líquida por produto."""
    docs = []
    for _, row in net_df.iterrows():
        texto = (
            f"A receita líquida do produto {row['product_name']} após as taxas do "
            f"Mercado Livre ({row['taxa_media_ml']*100:.0f}%) é de "
            f"R${row['receita_liquida']:,.2f} no período analisado. "
            f"O ticket líquido médio por unidade vendida é R${row['ticket_liquido']:.2f}. "
            f"O GMV bruto era R${row['gmv_bruto']:,.2f}."
        )
        docs.append({
            "id":   f"liquido_{row['product_id']}",
            "text": texto,
            "meta": {"tipo": "receita_liquida", "product_id": row["product_id"]},
        })
    return docs


def _doc_resumo_executivo(profile: dict, orders_path: str) -> dict:
    orders    = pd.read_csv(orders_path)
    gmv_total = orders["gmv"].sum()
    ticket    = gmv_total / orders["units_sold"].sum()
    prod_top  = orders.groupby("product_name")["units_sold"].sum().idxmax()

    # Inclui palavras-chave de ticket médio para melhorar recall
    text = (
        f"Resumo geral do vendedor Carlos Ferreira no Mercado Livre: "
        f"GMV total nos ultimos 18 meses foi de R${gmv_total:,.2f}. "
        f"Ticket medio por unidade vendida: R${ticket:.2f}. "
        f"Para aumentar o ticket medio, Carlos pode combinar produtos complementares "
        f"ou criar kits. Produto mais vendido: {prod_top}. "
        f"Reputacao: {profile['reputation_level']}. "
        f"Taxa de devolucao: {profile['return_rate_pct']}%. "
        f"Review medio: {profile['avg_review_score']} estrelas."
    )
    return {
        "id":   "resumo_executivo",
        "text": text,
        "meta": {"tipo": "resumo", "seller_id": profile["seller_id"]},
    }


# ──────────────────────────────────────────────────────────────
# Indexação
# ──────────────────────────────────────────────────────────────

def index_all(reset: bool = True) -> dict:
    """Indexa todos os documentos. Retorna contagens por tipo."""
    profile_path  = os.path.join(SYNTHETIC_DIR, "seller_profile.json")
    orders_path   = os.path.join(SYNTHETIC_DIR, "orders.csv")
    churn_path    = os.path.join(SPRINT2_DIR,   "churn_scores.csv")

    forecast_path = os.path.join(SPRINT3_DIR, "forecast_summary_corrigido.csv")
    if not os.path.exists(forecast_path):
        forecast_path = os.path.join(SPRINT2_DIR, "forecast_summary.csv")
        print(f"  Usando forecast original: {forecast_path}")

    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)

    forecast_df = pd.read_csv(forecast_path)
    churn_df    = pd.read_csv(churn_path)

    SPRINT4_DIR = os.path.join(BASE_DIR, "outputs", "sprint_4")
    conc_path   = os.path.join(SPRINT4_DIR, "revenue_concentration.csv")
    season_path = os.path.join(SPRINT4_DIR, "seasonality_index.csv")
    net_path    = os.path.join(SPRINT4_DIR, "net_revenue.csv")

    conc_df   = pd.read_csv(conc_path)   if os.path.exists(conc_path)   else pd.DataFrame()
    season_df = pd.read_csv(season_path) if os.path.exists(season_path) else pd.DataFrame()
    net_df    = pd.read_csv(net_path)    if os.path.exists(net_path)    else pd.DataFrame()

    all_docs = (
        _docs_perfil(profile)
        + _docs_tendencias(forecast_df)
        + _docs_churn(churn_df)
        + _docs_limitacoes()
        + [_doc_resumo_executivo(profile, orders_path)]
        + (_docs_concentracao(conc_df)   if not conc_df.empty   else [])
        + (_docs_sazonalidade(season_df) if not season_df.empty else [])
        + (_docs_receita_liquida(net_df) if not net_df.empty    else [])
    )

    embedding_fn = _get_embedding_fn()
    client       = _get_client()
    collection   = _get_or_reset_collection(client, embedding_fn, reset=reset)

    ids       = [d["id"]   for d in all_docs]
    documents = [d["text"] for d in all_docs]
    metadatas = [d["meta"] for d in all_docs]

    batch_size = 50
    for i in range(0, len(all_docs), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )

    counts: dict[str, int] = {}
    for d in all_docs:
        t = d["meta"]["tipo"]
        counts[t] = counts.get(t, 0) + 1

    total = sum(counts.values())
    print(f"\n  {total} documentos indexados (modelo: {EMBEDDING_MODEL}):")
    for tipo, n in sorted(counts.items()):
        print(f"    {tipo:12s}: {n}")

    return counts


# ──────────────────────────────────────────────────────────────
# Recuperação
# ──────────────────────────────────────────────────────────────

def ensure_indexed() -> bool:
    """
    Verifica se a collection tem documentos. Se estiver vazia,
    executa a indexação completa automaticamente.
    Retorna True se já estava indexado, False se re-indexou.
    """
    try:
        client   = _get_client()
        embed_fn = _get_embedding_fn()
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        if col.count() > 0:
            return True
        print("Collection vazia — re-indexando documentos...")
        index_all(reset=False)
        return False
    except Exception as e:
        print(f"Erro ao verificar indexação: {e}")
        return False


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """Busca semantica. Retorna lista de dicts: id, text, metadata, distance."""
    embedding_fn = _get_embedding_fn()
    client       = _get_client()
    collection   = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "id":       results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": round(results["distances"][0][i], 4),
        }
        for i in range(len(results["ids"][0]))
    ]


# ──────────────────────────────────────────────────────────────
# Runner com teste v2 (3 queries em PT-BR)
# ──────────────────────────────────────────────────────────────

def _format_hits(query: str, hits: list[dict]) -> str:
    sep  = "=" * 60
    dash = "-" * 60
    lines = [f'Query: "{query}"', sep, ""]
    for rank, h in enumerate(hits, 1):
        lines += [
            f"  #{rank}  [{h['metadata'].get('tipo', '?')}]  id={h['id']}",
            f"       distancia coseno: {h['distance']} (menor = mais similar)",
            f"       {h['text'][:180]}{'...' if len(h['text']) > 180 else ''}",
            "",
        ]
    lines.append(dash)
    return "\n".join(lines)


def run() -> dict:
    os.makedirs(SPRINT3_DIR, exist_ok=True)

    print(f"Modelo de embedding: {EMBEDDING_MODEL}")
    print("Indexando documentos no ChromaDB ...")
    counts = index_all(reset=True)

    # 3 queries de validacao em PT-BR
    test_queries = [
        "minhas vendas caíram",
        "risco de abandonar produto",
        "aumentar ticket médio",
    ]

    print("\nTestando recuperacao com 3 queries em PT-BR ...")
    output_blocks = [
        "# Teste de Recuperacao ChromaDB — v2",
        f"Modelo de embedding: {EMBEDDING_MODEL}",
        f"Collection: {COLLECTION_NAME}",
        "",
    ]

    for q in test_queries:
        hits = retrieve(q, top_k=3)
        block = _format_hits(q, hits)
        print("\n" + block)
        output_blocks.append(block + "\n")

    test_path = os.path.join(SPRINT3_DIR, "chroma_test_v2.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_blocks))
    print(f"\n  Resultado salvo em {test_path}")

    return counts


if __name__ == "__main__":
    run()
