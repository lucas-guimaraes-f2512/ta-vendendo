"""
Pipeline ETL — lê orders.csv e gera dados processados para os modelos.
Saída: data/processed/orders_clean.csv e data/processed/weekly_gmv.csv
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_and_clean() -> pd.DataFrame:
    path = os.path.join(SYNTHETIC_DIR, "orders.csv")
    df = pd.read_csv(path, parse_dates=["date"])

    # Remove nulos em colunas críticas
    critical = ["order_id", "date", "product_id", "units_sold", "unit_price", "gmv"]
    before = len(df)
    df = df.dropna(subset=critical)
    dropped = before - len(df)
    if dropped:
        print(f"  Removidas {dropped} linhas com nulos em colunas críticas")

    # platform_fee_pct já é decimal (0.14), não percentual
    df["revenue"] = df["units_sold"] * df["unit_price"] * (1 - df["platform_fee_pct"])
    df["is_return"] = df["return_flag"].astype(int)

    # Colunas temporais
    df["week"] = df["date"].dt.strftime("%Y-%W")
    df["month"] = df["date"].dt.strftime("%Y-%m")

    return df


def build_weekly_gmv(df: pd.DataFrame) -> pd.DataFrame:
    # ds = segunda-feira da semana (formato datetime para Prophet)
    df = df.copy()
    df["ds"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")

    weekly = (
        df.groupby(["ds", "product_id", "product_name"])
        .agg(
            y=("gmv", "sum"),
            units=("units_sold", "sum"),
            visits=("visit_count", "sum"),
            conversion_rate=("conversion_rate", "mean"),
            returns=("is_return", "sum"),
        )
        .reset_index()
        .sort_values(["product_id", "ds"])
    )
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    return weekly


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Carregando e limpando orders.csv ...")
    df = load_and_clean()

    orders_path = os.path.join(PROCESSED_DIR, "orders_clean.csv")
    df.to_csv(orders_path, index=False)
    print(f"  -> orders_clean.csv: {len(df):,} linhas salvas em {orders_path}")

    print("Gerando agregação semanal por produto ...")
    weekly = build_weekly_gmv(df)

    weekly_path = os.path.join(PROCESSED_DIR, "weekly_gmv.csv")
    weekly.to_csv(weekly_path, index=False)
    print(f"  -> weekly_gmv.csv: {len(weekly):,} linhas salvas em {weekly_path}")

    return df, weekly


if __name__ == "__main__":
    run()
