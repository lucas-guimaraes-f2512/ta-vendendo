"""
Business Insights — análise complementar ao churn e forecast.

Gera três arquivos em outputs/sprint_4/:
  1. revenue_concentration.csv  — concentração de receita por produto
  2. seasonality_index.csv      — índice de sazonalidade mensal
  3. net_revenue.csv            — receita líquida estimada por produto

Execute com: python src/models/business_insights.py
"""
import os
import pandas as pd
import numpy as np

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs", "sprint_4")

MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março",    4: "Abril",
    5: "Maio",    6: "Junho",     7: "Julho",     8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}


# ──────────────────────────────────────────────────────────────
# 1. Concentração de receita
# ──────────────────────────────────────────────────────────────

def revenue_concentration(gmv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Por produto: GMV total, participação % e classificação.
      crítico   : pct > 20 %
      relevante : pct > 10 %
      secundário : pct <= 10 %
    """
    agg = (
        gmv_df.groupby(["product_id", "product_name"], sort=False)["y"]
        .sum()
        .reset_index()
        .rename(columns={"y": "gmv_total_periodo"})
    )

    total_gmv = agg["gmv_total_periodo"].sum()
    agg["pct_receita"] = (agg["gmv_total_periodo"] / total_gmv * 100).round(2)

    def classifica(pct):
        if pct > 20:
            return "crítico"
        if pct > 10:
            return "relevante"
        return "secundário"

    agg["classificacao"] = agg["pct_receita"].apply(classifica)
    agg["gmv_total_periodo"] = agg["gmv_total_periodo"].round(2)

    return agg.sort_values("gmv_total_periodo", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 2. Sazonalidade histórica
# ──────────────────────────────────────────────────────────────

def seasonality_index(gmv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Por mês (1–12): GMV médio mensal, índice sazonal e classificação.
    Usa a mediana mensal como referência (robusta a picos de Black Friday).
      pico   : índice > 1.5
      normal : índice > 0.85
      baixo  : índice <= 0.85
    """
    df = gmv_df.copy()
    df["mes"]  = df["ds"].dt.month
    df["ano"]  = df["ds"].dt.year

    # GMV total por mês-ano (todos os produtos somados)
    mensal = (
        df.groupby(["ano", "mes"])["y"]
        .sum()
        .reset_index()
        .rename(columns={"y": "gmv_mes_ano"})
    )

    # Média de cada mês do calendário ao longo de todos os anos
    por_mes = (
        mensal.groupby("mes")["gmv_mes_ano"]
        .mean()
        .reset_index()
        .rename(columns={"gmv_mes_ano": "gmv_medio_mensal"})
    )

    # Mediana como referência — ignora extremos de Black Friday / Natal
    mediana_mensal = por_mes["gmv_medio_mensal"].median()
    por_mes["indice_sazonal"] = (por_mes["gmv_medio_mensal"] / mediana_mensal).round(4)
    por_mes["gmv_medio_mensal"] = por_mes["gmv_medio_mensal"].round(2)

    def classifica(idx):
        if idx > 1.5:
            return "pico"
        if idx > 0.85:
            return "normal"
        return "baixo"

    por_mes["classificacao"] = por_mes["indice_sazonal"].apply(classifica)
    por_mes["mes_nome"] = por_mes["mes"].map(MESES_PT)

    # Reordena colunas conforme spec
    return por_mes[["mes", "mes_nome", "gmv_medio_mensal", "indice_sazonal", "classificacao"]]


# ──────────────────────────────────────────────────────────────
# 3. Receita líquida estimada
# ──────────────────────────────────────────────────────────────

def net_revenue(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Por produto: GMV bruto, taxa média ML, receita líquida e ticket líquido.
    """
    agg = (
        orders_df.groupby(["product_id", "product_name"], sort=False)
        .agg(
            gmv_bruto=("gmv",              "sum"),
            taxa_media_ml=("platform_fee_pct", "mean"),
            units_total=("units_sold",     "sum"),
        )
        .reset_index()
    )

    agg["receita_liquida"] = (agg["gmv_bruto"] * (1 - agg["taxa_media_ml"])).round(2)
    agg["ticket_liquido"]  = (agg["receita_liquida"] / agg["units_total"]).round(2)
    agg["gmv_bruto"]       = agg["gmv_bruto"].round(2)
    agg["taxa_media_ml"]   = agg["taxa_media_ml"].round(4)

    return (
        agg[["product_id", "product_name", "gmv_bruto",
             "taxa_media_ml", "receita_liquida", "ticket_liquido"]]
        .sort_values("receita_liquida", ascending=False)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Carrega fontes
    gmv_path    = os.path.join(PROCESSED_DIR, "weekly_gmv.csv")
    orders_path = os.path.join(SYNTHETIC_DIR, "orders.csv")

    gmv_df    = pd.read_csv(gmv_path,    parse_dates=["ds"])
    orders_df = pd.read_csv(orders_path, parse_dates=["date"])

    print(f"  weekly_gmv  : {len(gmv_df)} linhas")
    print(f"  orders.csv  : {len(orders_df)} linhas")

    # ── 1. Concentração de receita ────────────────────────────
    print("\n[1] Concentracao de receita ...")
    rc = revenue_concentration(gmv_df)
    rc_path = os.path.join(OUTPUT_DIR, "revenue_concentration.csv")
    rc.to_csv(rc_path, index=False)

    col_w = max(len(n) for n in rc["product_name"]) + 2
    print(f"  {'Produto':<{col_w}} {'GMV Total':>14}  {'Pct %':>6}  Classificacao")
    print("  " + "-" * (col_w + 36))
    for _, r in rc.iterrows():
        print(f"  {r['product_name']:<{col_w}} "
              f"R$ {r['gmv_total_periodo']:>11,.2f}  "
              f"{r['pct_receita']:>5.1f}%  "
              f"{r['classificacao']}")
    print(f"\n  -> {rc_path}")

    # ── 2. Sazonalidade ───────────────────────────────────────
    print("\n[2] Indice de sazonalidade ...")
    si = seasonality_index(gmv_df)
    si_path = os.path.join(OUTPUT_DIR, "seasonality_index.csv")
    si.to_csv(si_path, index=False)

    print(f"  {'Mes':<12} {'GMV medio':>12}  {'Indice':>7}  Classificacao")
    print("  " + "-" * 46)
    for _, r in si.iterrows():
        print(f"  {r['mes_nome']:<12} "
              f"R$ {r['gmv_medio_mensal']:>9,.2f}  "
              f"{r['indice_sazonal']:>7.4f}  "
              f"{r['classificacao']}")
    print(f"\n  -> {si_path}")

    # ── 3. Receita líquida ────────────────────────────────────
    print("\n[3] Receita liquida estimada ...")
    nr = net_revenue(orders_df)
    nr_path = os.path.join(OUTPUT_DIR, "net_revenue.csv")
    nr.to_csv(nr_path, index=False)

    col_w = max(len(n) for n in nr["product_name"]) + 2
    print(f"  {'Produto':<{col_w}} {'GMV Bruto':>12}  {'Taxa ML':>7}  "
          f"{'Rec. Liq.':>12}  {'Ticket Liq.':>11}")
    print("  " + "-" * (col_w + 52))
    for _, r in nr.iterrows():
        print(f"  {r['product_name']:<{col_w}} "
              f"R$ {r['gmv_bruto']:>9,.2f}  "
              f"{r['taxa_media_ml']*100:>6.1f}%  "
              f"R$ {r['receita_liquida']:>9,.2f}  "
              f"R$ {r['ticket_liquido']:>8.2f}")
    print(f"\n  -> {nr_path}")

    print("\nBUSINESS INSIGHTS concluido.")
    return {"revenue_concentration": rc, "seasonality_index": si, "net_revenue": nr}


if __name__ == "__main__":
    run()
