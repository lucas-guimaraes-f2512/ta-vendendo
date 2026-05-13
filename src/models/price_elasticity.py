"""
Elasticidade de preço via regressão log-log e correlação.
Analisa os 3 produtos com maior volume de vendas.
Saída: outputs/sprint_2/elasticity_report.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sprint_2")


def _estimate_elasticity(df_prod: pd.DataFrame) -> dict:
    """
    Regressão log-log: ln(conversion_rate) ~ β * ln(unit_price)
    β é o coeficiente de elasticidade.
    """
    data = df_prod[["unit_price", "conversion_rate", "units_sold"]].dropna()
    data = data[(data["unit_price"] > 0) & (data["conversion_rate"] > 0)]

    if len(data) < 20:
        return {"elasticity": None, "r2": None}

    log_price = np.log(data["unit_price"].values).reshape(-1, 1)
    log_conv = np.log(data["conversion_rate"].values)

    model = LinearRegression()
    model.fit(log_price, log_conv)
    r2 = model.score(log_price, log_conv)

    return {
        "elasticity": round(model.coef_[0], 3),
        "r2": round(r2, 3),
    }


def _build_recommendation(elasticity: float, current_price: float, prod_name: str) -> tuple[str, str]:
    """Retorna (recommended_action, expected_impact)."""
    if elasticity is None:
        return "Dados insuficientes para análise", "–"

    price_change_pct = 10  # simula redução de 10%
    expected_conv_change = elasticity * (-price_change_pct)

    if elasticity < -0.5:
        action = f"Reduzir preço em {price_change_pct}% pode aumentar conversão"
        impact = f"↑ {abs(expected_conv_change):.1f}% na conversão (elasticidade = {elasticity:.2f})"
    elif -0.5 <= elasticity <= 0.1:
        action = "Preço atual já está no ponto ótimo observado"
        impact = f"Elasticidade baixa ({elasticity:.2f}): variações de preço têm pouco efeito"
    else:
        action = "Aumentar preço pode não reduzir conversão significativamente"
        impact = f"Correlação positiva incomum ({elasticity:.2f}): produto de alto valor percebido"

    return action, impact


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    orders_path = os.path.join(PROCESSED_DIR, "orders_clean.csv")
    df = pd.read_csv(orders_path, parse_dates=["date"])

    # Top 3 produtos por volume total
    top3 = (
        df.groupby(["product_id", "product_name"])["units_sold"]
        .sum()
        .nlargest(3)
        .reset_index()
    )

    print(f"Top 3 produtos por volume: {top3['product_name'].tolist()}")

    rows = []
    for _, row in top3.iterrows():
        pid = row["product_id"]
        pname = row["product_name"]
        df_prod = df[df["product_id"] == pid]

        current_price = df_prod["unit_price"].iloc[-30:].mean()
        result = _estimate_elasticity(df_prod)
        elasticity = result["elasticity"]
        r2 = result["r2"]

        action, impact = _build_recommendation(elasticity, current_price, pname)

        rows.append({
            "product_id": pid,
            "product_name": pname,
            "elasticity_coefficient": elasticity,
            "r2_score": r2,
            "current_price": round(current_price, 2),
            "recommended_action": action,
            "expected_impact": impact,
        })

        print(f"\n  [{pid}] {pname}")
        print(f"    Preço atual: R$ {current_price:.2f}")
        print(f"    Elasticidade: {elasticity}  R²: {r2}")
        print(f"    Recomendação: {action}")
        print(f"    Impacto esperado: {impact}")

    report_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "elasticity_report.csv")
    report_df.to_csv(out_path, index=False)
    print(f"\n  -> elasticity_report.csv salvo em {out_path}")

    return report_df


if __name__ == "__main__":
    run()
