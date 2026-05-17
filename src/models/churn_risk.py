"""
Score de risco de churn por produto — modelo ponderado multi-feature.
Lê data/synthetic/orders.csv e filtra os últimos 90 dias disponíveis.

Pesos:
  0.40 × queda_30   (queda de GMV nos últimos 30d vs. 30d anteriores)
  0.25 × queda_60   (tendência de médio prazo)
  0.20 × return_rate normalizada
  0.15 × review normalizado

Saída: outputs/sprint_2/churn_scores.csv + churn_chart.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")
SPRINT2_DIR   = os.path.join(BASE_DIR, "outputs", "sprint_2")
SPRINT3_DIR   = os.path.join(BASE_DIR, "outputs", "sprint_3")


# ──────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────

def _score_products(orders: pd.DataFrame, forecast: pd.DataFrame | None) -> pd.DataFrame:
    max_date  = orders["date"].max()
    cutoff_30 = max_date - pd.Timedelta(days=30)
    cutoff_60 = max_date - pd.Timedelta(days=60)
    cutoff_90 = max_date - pd.Timedelta(days=90)

    gmv_total_all = float(orders["gmv"].sum())

    products = (
        orders[["product_id", "product_name"]]
        .drop_duplicates()
        .sort_values("product_id")
    )

    records = []

    for _, prod in products.iterrows():
        pid   = prod["product_id"]
        pname = prod["product_name"]
        p     = orders[orders["product_id"] == pid]

        # ── janelas ──────────────────────────────────────────
        periodo_30 = p[p["date"] > cutoff_30]                                    # últimos 30d
        periodo_60 = p[(p["date"] > cutoff_60) & (p["date"] <= cutoff_30)]       # 31–60d atrás
        periodo_90 = p[p["date"] > cutoff_90]                                    # últimos 90d

        gmv_30 = float(periodo_30["gmv"].sum()) if not periodo_30.empty else 0.0
        gmv_60 = float(periodo_60["gmv"].sum()) if not periodo_60.empty else 0.0

        # ── features de GMV ──────────────────────────────────
        if gmv_60 > 0:
            queda_30 = max(0.0, (gmv_60 - gmv_30) / gmv_60)
        else:
            queda_30 = 0.0

        queda_60 = max(0.0, (gmv_60 - gmv_30) / (gmv_60 + gmv_30 + 1e-9))

        # gmv_var_30d: últimos 30d vs. 31–60d atrás
        gmv_var_30d = ((gmv_30 - gmv_60) / (gmv_60 + 1e-9) * 100)

        # gmv_var_60d: janela 31–60d vs. janela 61–90d
        periodo_prev = p[(p["date"] > (max_date - pd.Timedelta(days=90))) &
                         (p["date"] <= (max_date - pd.Timedelta(days=60)))]
        gmv_prev = float(periodo_prev["gmv"].sum()) if not periodo_prev.empty else 0.0
        gmv_var_60d = ((gmv_60 - gmv_prev) / (gmv_prev + 1e-9) * 100)

        # ── review ───────────────────────────────────────────
        if not periodo_90.empty:
            review_medio = float(periodo_90["review_score"].mean())
        else:
            review_medio = 5.0
        review_norm = max(0.0, (5.0 - review_medio) / 5.0)

        # ── devolução ────────────────────────────────────────
        pedidos_90 = len(periodo_90)
        devolucoes = int(periodo_90["return_flag"].sum()) if not periodo_90.empty else 0
        return_rate = devolucoes / pedidos_90 if pedidos_90 > 0 else 0.0

        # ── score final ──────────────────────────────────────
        c_queda30  = 0.40 * queda_30
        c_queda60  = 0.25 * queda_60
        c_retorno  = 0.20 * min(1.0, return_rate * 5)
        c_review   = 0.15 * review_norm

        score = round(min(1.0, max(0.0, c_queda30 + c_queda60 + c_retorno + c_review)), 4)

        # ── proteção para produtos âncora (> 20% do GMV total) ──────────
        gmv_total_prod = float(p["gmv"].sum())
        pct_receita    = gmv_total_prod / (gmv_total_all + 1e-9)
        if pct_receita > 0.20:
            c_queda30 *= 0.60
            c_queda60 *= 0.60
            score = round(min(1.0, max(0.0, c_queda30 + c_queda60 + c_retorno + c_review)), 4)

        # ── nível de risco ───────────────────────────────────
        if score >= 0.40:
            risk_level = "High"
        elif score >= 0.20:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # ── sinal principal ──────────────────────────────────
        contribs = [
            (c_queda30, f"Queda de GMV de {queda_30*100:.0f}% nos últimos 30 dias"),
            (c_queda60, f"Queda acumulada de GMV de {queda_60*100:.0f}% no período de 60 dias"),
            (c_retorno, f"Taxa de devolução elevada ({return_rate*100:.0f}%) nos últimos 90 dias"),
            (c_review,  f"Review médio baixo ({review_medio:.1f} estrelas) nos últimos 90 dias"),
        ]
        top_contrib, main_signal = max(contribs, key=lambda t: t[0])
        if top_contrib == 0.0:
            main_signal = "Tendência estável — risco baixo"

        records.append({
            "product_id":         pid,
            "product_name":       pname,
            "churn_risk_score":   score,
            "risk_level":         risk_level,
            "main_signal":        main_signal,
            "review_score_medio": round(review_medio, 2),
            "return_rate_90d":    round(return_rate, 4),
            "gmv_var_30d":        round(gmv_var_30d, 1),
            "gmv_var_60d":        round(gmv_var_60d, 1),
            "alerta_divergencia": False,
        })

    result = pd.DataFrame(records)

    # ── divergência churn vs. forecast ───────────────────────
    if forecast is not None and not forecast.empty:
        for idx, row in result.iterrows():
            fc = forecast[forecast["product_id"] == row["product_id"]]
            if fc.empty:
                continue
            pct = float(fc.iloc[0]["pct_change"])
            if pct < -50 and row["risk_level"] != "High":
                result.at[idx, "alerta_divergencia"] = True
                result.at[idx, "main_signal"] = (
                    row["main_signal"]
                    + f" | Atenção: forecast indica queda de {pct:.0f}%"
                )

    return result.sort_values("churn_risk_score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Gráfico
# ──────────────────────────────────────────────────────────────

def _plot_churn(scores: pd.DataFrame, out_path: str) -> None:
    COLOR_MAP = {"Low": "#66bb6a", "Medium": "#ffa726", "High": "#ef5350"}

    df     = scores.sort_values("churn_risk_score")
    colors = [COLOR_MAP[r] for r in df["risk_level"]]
    labels = [n if len(n) <= 28 else n[:26] + "…" for n in df["product_name"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.barh(labels, df["churn_risk_score"], color=colors, height=0.55)
    for bar, sc, lvl in zip(bars, df["churn_risk_score"], df["risk_level"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{sc:.3f}  {lvl}",
            va="center", ha="left", color="white", fontsize=9,
        )

    ax.set_xlim(0, 1.25)
    ax.set_xlabel(
        "Churn Risk Score (0 = sem risco · 1 = risco máximo)",
        color="#b0b8c1", fontsize=10,
    )
    ax.set_title("Risco de Churn por Produto", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#b0b8c1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")
    ax.grid(axis="x", color="#2a2d3e", linewidth=0.6)
    ax.axvline(0.40, color="#ef5350", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(0.20, color="#ffa726", linewidth=1, linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#66bb6a", label="Low  (< 0.20)"),
            Patch(facecolor="#ffa726", label="Medium (0.20–0.40)"),
            Patch(facecolor="#ef5350", label="High  (>= 0.40)"),
        ],
        facecolor="#1a1d2e", labelcolor="white", fontsize=9, loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    os.makedirs(SPRINT2_DIR, exist_ok=True)

    orders_path   = os.path.join(SYNTHETIC_DIR, "orders.csv")
    forecast_path = os.path.join(SPRINT3_DIR,   "forecast_summary_corrigido.csv")

    orders = pd.read_csv(orders_path, parse_dates=["date"])
    print(f"  {len(orders)} pedidos carregados")
    print(f"  Intervalo de datas: {orders['date'].min().date()} a {orders['date'].max().date()}")

    if os.path.exists(forecast_path):
        forecast = pd.read_csv(forecast_path)
        print(f"  Forecast carregado: {len(forecast)} produtos")
    else:
        forecast = None
        print("  Forecast não encontrado — divergência não será detectada")

    print("\nCalculando scores ...")
    scores = _score_products(orders, forecast)

    # CSV
    csv_path = os.path.join(SPRINT2_DIR, "churn_scores.csv")
    scores.to_csv(csv_path, index=False)
    print(f"  -> churn_scores.csv salvo ({len(scores)} produtos)")

    # Gráfico
    chart_path = os.path.join(SPRINT2_DIR, "churn_chart.png")
    _plot_churn(scores, chart_path)
    print(f"  -> churn_chart.png salvo")

    # Resumo
    col_w = max(len(n) for n in scores["product_name"]) + 2
    print(f"\n  {'Produto':<{col_w}} {'Score':>6}  {'Nível':<8}  {'Diverg.':<8}  Sinal")
    print("  " + "-" * (col_w + 62))
    for _, r in scores.iterrows():
        div    = "SIM" if r["alerta_divergencia"] else "nao"
        sinal  = r["main_signal"][:65] + ("…" if len(r["main_signal"]) > 65 else "")
        print(f"  {r['product_name']:<{col_w}} "
              f"{r['churn_risk_score']:>6.3f}  "
              f"{r['risk_level']:<8}  "
              f"{div:<8}  "
              f"{sinal}")

    return scores


if __name__ == "__main__":
    run()
