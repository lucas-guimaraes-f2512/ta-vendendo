"""
Gerador de dados sintéticos para o projeto "Tá vendendo?"
Gera 18 meses de pedidos e perfil do vendedor Carlos.
"""

import json
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# --- Configurações base ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 6, 30)

# --- Catálogo de produtos ---
PRODUCTS = [
    {"id": "PROD001", "name": "Fone de Ouvido Bluetooth JBL T510",   "category": "Audio",        "base_price": 189.90, "base_daily_units": 3.2, "return_rate": 0.025, "avg_review": 4.7, "review_std": 0.3},
    {"id": "PROD002", "name": "Carregador Turbo 65W USB-C",           "category": "Carregadores", "base_price": 89.90,  "base_daily_units": 5.1, "return_rate": 0.055, "avg_review": 3.9, "review_std": 0.6},
    {"id": "PROD003", "name": "Cabo USB-C para USB-C 2m",             "category": "Cabos",        "base_price": 29.90,  "base_daily_units": 7.8, "return_rate": 0.030, "avg_review": 4.5, "review_std": 0.3},
    {"id": "PROD004", "name": "Suporte Veicular para Celular",         "category": "Acessórios",   "base_price": 49.90,  "base_daily_units": 4.5, "return_rate": 0.028, "avg_review": 4.4, "review_std": 0.4},
    {"id": "PROD005", "name": "Película Vidro Temperado iPhone 15",   "category": "Proteção",     "base_price": 24.90,  "base_daily_units": 6.2, "return_rate": 0.018, "avg_review": 4.6, "review_std": 0.3},
    {"id": "PROD006", "name": "Hub USB 4 portas 3.0",                 "category": "Conectividade","base_price": 79.90,  "base_daily_units": 2.8, "return_rate": 0.048, "avg_review": 4.1, "review_std": 0.5},
    {"id": "PROD007", "name": "Fone In-Ear Com Fio Samsung AKG",      "category": "Audio",        "base_price": 59.90,  "base_daily_units": 1.5, "return_rate": 0.022, "avg_review": 4.3, "review_std": 0.4},  # tendência de queda
    {"id": "PROD008", "name": "Cabo Lightning para USB 1m (Apple)",   "category": "Cabos",        "base_price": 39.90,  "base_daily_units": 1.8, "return_rate": 0.060, "avg_review": 3.7, "review_std": 0.7},  # tendência de queda
]

# --- Fatores sazonais por mês ---
SEASONAL_FACTORS = {
    1: 0.85,   # Janeiro: pós-festas, queda
    2: 0.72,   # Fevereiro: carnaval, menor mês
    3: 0.78,   # Março: ainda baixo
    4: 0.88,   # Abril: recuperação
    5: 0.92,   # Maio: dia das mães
    6: 0.95,   # Junho: dia dos namorados
    7: 0.90,   # Julho: férias escolares
    8: 0.93,   # Agosto: estável
    9: 0.97,   # Setembro: aquecendo
    10: 1.05,  # Outubro: pré-BF
    11: 1.85,  # Novembro: Black Friday
    12: 1.60,  # Dezembro: Natal
}

# --- Fatores sazonais por categoria ---
CATEGORY_SEASONAL_FACTORS = {
    "Audio": {
        1: 0.80, 2: 0.65, 3: 0.70, 4: 0.85, 5: 1.10,
        6: 0.90, 7: 0.95, 8: 1.15,
        9: 0.92, 10: 1.05, 11: 1.90, 12: 1.75,
    },
    "Carregadores": {
        1: 0.85, 2: 0.70, 3: 0.80, 4: 0.90, 5: 0.95,
        6: 0.95, 7: 0.90, 8: 0.92, 9: 0.98,
        10: 1.05, 11: 1.80, 12: 1.65,
    },
    "Cabos": {
        1: 0.88, 2: 0.75, 3: 0.82, 4: 0.90, 5: 0.92,
        6: 0.93, 7: 0.88, 8: 0.90, 9: 0.95,
        10: 1.02, 11: 1.70, 12: 1.50,
    },
    "Proteção": {
        1: 0.80, 2: 0.65, 3: 0.72, 4: 0.85, 5: 0.88,
        6: 0.85, 7: 0.88, 8: 0.90, 9: 1.45,
        10: 1.10, 11: 1.85, 12: 1.40,
    },
    "Acessórios": {
        1: 0.82, 2: 0.68, 3: 0.75, 4: 0.88, 5: 0.92,
        6: 1.05, 7: 0.90, 8: 0.93, 9: 0.95,
        10: 1.05, 11: 1.75, 12: 1.60,
    },
    "Conectividade": {
        1: 0.90, 2: 0.78, 3: 1.05, 4: 0.92, 5: 0.93,
        6: 0.95, 7: 0.85, 8: 1.08, 9: 1.00,
        10: 1.05, 11: 1.65, 12: 1.45,
    },
}

PLATFORM_FEE_PCT = 0.14  # 14% fee médio Mercado Livre


def seasonal_multiplier(date: datetime, product_id: str) -> float:
    category = next(p["category"] for p in PRODUCTS if p["id"] == product_id)
    base_factors = CATEGORY_SEASONAL_FACTORS.get(category, SEASONAL_FACTORS)
    month_factor = base_factors[date.month]

    # Black Friday: semana específica novembro 2024 (semana 4)
    if date.year == 2024 and date.month == 11 and 25 <= date.day <= 30:
        month_factor *= 1.5

    # Fim de semana tem leve queda em eletrônicos B2C
    if date.weekday() >= 5:
        month_factor *= 0.85

    # PROD007 e PROD008: tendência de queda nos últimos 3 meses (abr-jun 2025)
    if product_id in ("PROD007", "PROD008"):
        if date >= datetime(2025, 4, 1):
            months_into_decline = (date.year - 2025) * 12 + (date.month - 4)
            month_factor *= max(0.3, 1.0 - 0.20 * months_into_decline)

    return month_factor


def generate_orders() -> pd.DataFrame:
    rows = []
    order_id = 1000

    current = START_DATE
    while current <= END_DATE:
        for prod in PRODUCTS:
            multiplier = seasonal_multiplier(current, prod["id"])
            base_units = prod["base_daily_units"] * multiplier

            # Ruído aleatório diário
            units = max(0, int(np.random.poisson(base_units)))

            # MELHORIA 5 — Ruptura de estoque PROD006 (fev-abr 2025)
            if prod["id"] == "PROD006" and datetime(2025, 2, 15) <= current <= datetime(2025, 4, 10):
                units = 0  # sem estoque — não gera pedido
                current += timedelta(days=1)
                continue

            if units == 0:
                current += timedelta(days=1)
                continue

            # Variação de preço: ±8% ao redor do preço base
            price_variation = np.random.uniform(-0.08, 0.08)
            unit_price = round(prod["base_price"] * (1 + price_variation), 2)

            # MELHORIA 4 — Erosão competitiva: -0.3% ao mês a partir do 4º mês
            months_elapsed = (current.year - START_DATE.year) * 12 + (current.month - START_DATE.month)
            if months_elapsed > 3:
                competitive_decay = max(0.82, 1.0 - 0.003 * (months_elapsed - 3))
                unit_price = round(unit_price * competitive_decay, 2)

            # MELHORIA 4 — Desconto explícito Black Friday: 12-18%
            if current.year == 2024 and current.month == 11 and 25 <= current.day <= 30:
                bf_discount = np.random.uniform(0.12, 0.18)
                unit_price = round(unit_price * (1 - bf_discount), 2)

            gmv = round(units * unit_price, 2)

            # Visitas: conversão típica de 2-6%
            conversion_rate = round(np.random.uniform(0.02, 0.06), 4)
            visit_count = max(units, int(units / conversion_rate))

            # MELHORIA 2 — Devolução por taxa específica do produto
            return_flag = 1 if random.random() < prod["return_rate"] else 0

            # MELHORIA 3 — Review correlacionado ao produto e ao retorno
            if return_flag:
                review_score = round(max(1.0, min(5.0,
                    np.random.normal(1.8, 0.6)
                )), 1)
            else:
                review_score = round(max(3.0, min(5.0,
                    np.random.normal(prod["avg_review"], prod["review_std"])
                )), 1)

            rows.append({
                "order_id": order_id,
                "date": current.strftime("%Y-%m-%d"),
                "product_id": prod["id"],
                "product_name": prod["name"],
                "category": prod["category"],
                "units_sold": units,
                "unit_price": unit_price,
                "platform_fee_pct": PLATFORM_FEE_PCT,
                "gmv": gmv,
                "visit_count": visit_count,
                "conversion_rate": conversion_rate,
                "return_flag": return_flag,
                "review_score": review_score,
            })
            order_id += 1

        current += timedelta(days=1)

    return pd.DataFrame(rows)


def generate_seller_profile() -> dict:
    active_products = []
    for prod in PRODUCTS:
        stock = random.randint(15, 200)
        active_products.append({
            "id": prod["id"],
            "name": prod["name"],
            "category": prod["category"],
            "current_price": prod["base_price"],
            "stock": stock,
        })

    return {
        "seller_id": "ML-CARLOS-2020",
        "name": "Carlos Ferreira",
        "category": "Eletrônicos e Acessórios",
        "platform": "Mercado Livre",
        "start_date": "2020-03-15",
        "active_products": active_products,
        "avg_monthly_gmv": 28500.00,
        "health_score": 82,
        "churn_risk_score": 0.18,
        "reputation_level": "MercadoLíder Gold",
        "fulfillment_type": "Full",
        "avg_shipping_days": 1.8,
        "return_rate_pct": 3.1,
        "avg_review_score": 4.7,
    }


def compute_stats(df: pd.DataFrame) -> dict:
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    monthly_gmv = df.groupby("month")["gmv"].sum()
    peak_month = str(monthly_gmv.idxmax())
    low_month = str(monthly_gmv.idxmin())

    best_product = df.groupby("product_name")["units_sold"].sum().idxmax()
    ticket_medio = round(df["gmv"].sum() / df["units_sold"].sum(), 2)

    return {
        "total_rows": len(df),
        "gmv_total": round(df["gmv"].sum(), 2),
        "ticket_medio": ticket_medio,
        "produto_mais_vendido": best_product,
        "mes_pico": peak_month,
        "mes_queda": low_month,
    }


def main():
    print("Gerando orders.csv ...")
    orders_df = generate_orders()
    orders_path = os.path.join(OUTPUT_DIR, "orders.csv")
    orders_df.to_csv(orders_path, index=False, encoding="utf-8")
    print(f"  -> {len(orders_df)} linhas salvas em {orders_path}")

    print("Gerando seller_profile.json ...")
    profile = generate_seller_profile()
    profile_path = os.path.join(OUTPUT_DIR, "seller_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"  -> Perfil salvo em {profile_path}")

    stats = compute_stats(orders_df)

    # --- Sprint 2 log ---
    log_dir = os.path.join(BASE_DIR, "outputs", "sprint_2")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "sprint_2_log.md")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"""# Sprint 2 — Log de Execução

## Informações gerais
- **Data e hora de execução:** {now}
- **Script:** `src/data/generate_synthetic.py`

---

## Arquivos gerados

| Arquivo | Caminho completo | Linhas/Itens |
|---|---|---|
| `orders.csv` | `{orders_path}` | {stats['total_rows']:,} linhas |
| `seller_profile.json` | `{profile_path}` | 1 perfil (8 produtos ativos) |

---

## Estatísticas dos dados sintéticos (orders.csv)

| Métrica | Valor |
|---|---|
| **GMV total (18 meses)** | R$ {stats['gmv_total']:,.2f} |
| **Ticket médio por unidade** | R$ {stats['ticket_medio']:,.2f} |
| **Produto mais vendido** | {stats['produto_mais_vendido']} |
| **Mês de pico** | {stats['mes_pico']} |
| **Mês de menor volume** | {stats['mes_queda']} |
| **Período coberto** | Janeiro 2024 a Junho 2025 (18 meses) |
| **Produtos simulados** | 8 SKUs de eletrônicos |
| **Produtos com tendência de queda** | PROD007 (Fone In-Ear Com Fio Samsung AKG), PROD008 (Cabo Lightning Apple) |

---

## Sazonalidade aplicada

- **Fevereiro/Março:** fator de queda (0.72–0.78) — menor período do ano
- **Novembro:** Black Friday com pico de 1.85× + bônus de 1.5× na semana de 25–30/nov
- **Dezembro:** fator 1.60× — Natal
- **PROD007 e PROD008:** queda progressiva de 20%/mês a partir de abril 2025 (simula churn)

---

## Próximos passos (Sprint 3)

1. Rodar `src/models/sales_forecast.py` para gerar previsões com Prophet
2. Rodar `src/models/churn_risk.py` para scorar risco de abandono por SKU
3. Rodar `src/models/price_elasticity.py` para estimar elasticidade de preço
"""

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    print(f"\nLog salvo em {log_path}")
    print("\n=== RESUMO ===")
    print(f"  Linhas geradas (orders.csv): {stats['total_rows']:,}")
    print(f"  GMV total: R$ {stats['gmv_total']:,.2f}")
    print(f"  Ticket médio: R$ {stats['ticket_medio']:,.2f}")
    print(f"  Produto mais vendido: {stats['produto_mais_vendido']}")
    print(f"  Mês de pico: {stats['mes_pico']}")
    print(f"  Mês de menor volume: {stats['mes_queda']}")


if __name__ == "__main__":
    main()
