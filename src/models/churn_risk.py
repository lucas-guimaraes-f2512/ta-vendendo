"""
Score de risco de churn por produto usando regras baseadas em features de tendência.
Usa os últimos 90 dias disponíveis no dataset.
Saída: outputs/sprint_2/churn_scores.csv + churn_chart.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sprint_2")


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    max_date = df["date"].max()
    cutoff_30 = max_date - pd.Timedelta(days=30)
    cutoff_60 = max_date - pd.Timedelta(days=60)
    cutoff_90 = max_date - pd.Timedelta(days=90)

    products = df[["product_id", "product_name"]].drop_duplicates()
    rows = []

    for _, row in products.iterrows():
        pid = row["product_id"]
        pname = row["product_name"]
        p = df[df["product_id"] == pid]

        def gmv_window(start, end):
            return p[(p["date"] > start) & (p["date"] <= end)]["gmv"].sum()

        gmv_30 = gmv_window(cutoff_30, max_date)
        gmv_60 = gmv_window(cutoff_60, cutoff_30)
        gmv_90 = gmv_window(cutoff_90, cutoff_60)

        def safe_pct(a, b):
            return (a - b) / (b + 1e-9) * 100

        trend_30_60 = safe_pct(gmv_30, gmv_60)
        trend_60_90 = safe_pct(gmv_60, gmv_90)

        last_30 = p[p["date"] > cutoff_30]
        avg_conv = last_30["conversion_rate"].mean() if not last_30.empty else 0
        return_rate = last_30["return_flag"].mean() if not last_30.empty else 0
        avg_review = last_30["review_score"].mean() if not last_30.empty else 5.0

        rows.append({
            "product_id": pid,
            "product_name": pname,
            "gmv_last_30d": round(gmv_30, 2),
            "gmv_last_60d": round(gmv_60, 2),
            "gmv_last_90d": round(gmv_90, 2),
            "gmv_trend_30_60": round(trend_30_60, 1),
            "gmv_trend_60_90": round(trend_60_90, 1),
            "avg_conversion_last_30d": round(avg_conv, 4),
            "return_rate_last_30d": round(return_rate, 4),
            "avg_review_last_30d": round(avg_review, 2),
        })

    return pd.DataFrame(rows)


def _score_churn(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Regras de scoring (0.0–1.0):
    - Duas quedas consecutivas de GMV > 15%: +0.45
    - Uma queda de GMV > 15%:                +0.20
    - GMV_30d < 50% do GMV_90d:              +0.20
    - Taxa de retorno > 5%:                  +0.10
    - Review médio < 3.5:                    +0.10
    - Conversão < 2%:                        +0.10
    Score é limitado a 1.0.
    """
    scores = []

    for _, r in feat.iterrows():
        score = 0.0
        signals = []

        d30_60 = r["gmv_trend_30_60"]
        d60_90 = r["gmv_trend_60_90"]

        if d30_60 < -15 and d60_90 < -15:
            score += 0.45
            signals.append(f"queda consecutiva de GMV ({d60_90:+.0f}% e {d30_60:+.0f}%)")
        elif d30_60 < -15:
            score += 0.20
            signals.append(f"queda de GMV nos últimos 30 dias ({d30_60:+.0f}%)")
        elif d60_90 < -15:
            score += 0.15
            signals.append(f"queda de GMV de 30–60 dias atrás ({d60_90:+.0f}%)")

        gmv_90 = r["gmv_last_90d"]
        gmv_30 = r["gmv_last_30d"]
        if gmv_90 > 0 and gmv_30 < gmv_90 * 0.50:
            score += 0.20
            signals.append("GMV atual abaixo de 50% do nível de 90 dias atrás")

        if r["return_rate_last_30d"] > 0.05:
            score += 0.10
            signals.append(f"taxa de devolução elevada ({r['return_rate_last_30d']*100:.1f}%)")

        if r["avg_review_last_30d"] < 3.5:
            score += 0.10
            signals.append(f"review médio baixo ({r['avg_review_last_30d']:.1f}★)")

        if r["avg_conversion_last_30d"] < 0.02:
            score += 0.10
            signals.append(f"conversão abaixo de 2% ({r['avg_conversion_last_30d']*100:.2f}%)")

        score = min(round(score, 2), 1.0)

        if score >= 0.50:
            risk_level = "High"
        elif score >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        main_signal = signals[0] if signals else "Sem sinais de alerta identificados"

        scores.append({
            "product_id": r["product_id"],
            "product_name": r["product_name"],
            "churn_risk_score": score,
            "risk_level": risk_level,
            "main_signal": main_signal,
        })

    return pd.DataFrame(scores).sort_values("churn_risk_score", ascending=False)


def _plot_churn(scores: pd.DataFrame, out_path: str):
    COLOR_MAP = {"Low": "#66bb6a", "Medium": "#ffa726", "High": "#ef5350"}

    df = scores.sort_values("churn_risk_score")
    colors = [COLOR_MAP[r] for r in df["risk_level"]]
    labels = [n if len(n) <= 28 else n[:26] + "…" for n in df["product_name"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.barh(labels, df["churn_risk_score"], color=colors, height=0.55)

    for bar, score, level in zip(bars, df["churn_risk_score"], df["risk_level"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}  {level}", va="center", ha="left",
                color="white", fontsize=9)

    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Churn Risk Score (0 = sem risco · 1 = risco máximo)",
                  color="#b0b8c1", fontsize=10)
    ax.set_title("Risco de Churn por Produto", color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#b0b8c1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")
    ax.grid(axis="x", color="#2a2d3e", linewidth=0.6)
    ax.axvline(0.50, color="#ef5350", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(0.25, color="#ffa726", linewidth=1, linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#66bb6a", label="Low (< 0.25)"),
        Patch(facecolor="#ffa726", label="Medium (0.25–0.50)"),
        Patch(facecolor="#ef5350", label="High (≥ 0.50)"),
    ]
    ax.legend(handles=legend_elements, facecolor="#1a1d2e", labelcolor="white", fontsize=9,
              loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    orders_path = os.path.join(PROCESSED_DIR, "orders_clean.csv")
    df = pd.read_csv(orders_path, parse_dates=["date"])

    print("Calculando features de churn ...")
    features = _compute_features(df)

    print("Aplicando regras de scoring ...")
    scores = _score_churn(features)

    csv_path = os.path.join(OUTPUT_DIR, "churn_scores.csv")
    scores.to_csv(csv_path, index=False)
    print(f"  -> churn_scores.csv salvo em {csv_path}")

    chart_path = os.path.join(OUTPUT_DIR, "churn_chart.png")
    _plot_churn(scores, chart_path)
    print(f"  -> churn_chart.png salvo em {chart_path}")

    print("\n  Resultado por produto:")
    for _, r in scores.iterrows():
        print(f"  [{r['product_id']}] {r['risk_level']:6s}  score={r['churn_risk_score']:.2f}  "
              f"— {r['main_signal']}")

    return scores


if __name__ == "__main__":
    run()
