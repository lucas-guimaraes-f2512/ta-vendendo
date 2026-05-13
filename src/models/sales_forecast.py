"""
Previsão de tendências de vendas com Prophet.
  run()           -> treino simples, outputs em outputs/sprint_2/
  run_corrected() -> com feriados BR + winsorização, outputs em outputs/sprint_3/
"""
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SPRINT2_DIR  = os.path.join(BASE_DIR, "outputs", "sprint_2")
SPRINT3_DIR  = os.path.join(BASE_DIR, "outputs", "sprint_3")

TRAIN_CUTOFF  = pd.Timestamp("2025-03-31")
FORECAST_WEEKS = 12


# ──────────────────────────────────────────────────────────────
# Feriados brasileiros relevantes (alinhados à segunda da semana)
# ──────────────────────────────────────────────────────────────

def _monday_of(date: pd.Timestamp) -> pd.Timestamp:
    """Retorna a segunda-feira da semana de 'date'."""
    return date - pd.Timedelta(days=date.dayofweek)


def _build_holidays() -> pd.DataFrame:
    """
    Cria DataFrame de feriados para o Prophet.
    'ds' é a segunda-feira da semana do feriado (para casar com weekly_gmv).
    """
    events = [
        # Black Friday: última sexta de novembro
        ("Black Friday", pd.Timestamp("2024-11-29")),  # sexta -> seg 25/nov
        ("Black Friday", pd.Timestamp("2025-11-28")),  # sexta -> seg 24/nov
        # Natal: 25/dez
        ("Natal",        pd.Timestamp("2024-12-25")),  # quarta -> seg 23/dez
        ("Natal",        pd.Timestamp("2025-12-25")),  # quinta -> seg 22/dez
        # Dias das Mães (2.° domingo de maio)
        ("Dia das Maes", pd.Timestamp("2024-05-12")),  # dom -> seg 06/mai
        ("Dia das Maes", pd.Timestamp("2025-05-11")),  # dom -> seg 05/mai
    ]

    rows = []
    for name, date in events:
        rows.append({
            "holiday": name,
            "ds": _monday_of(date),
            "lower_window": 0,
            "upper_window": 0,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Winsorização por série de produto
# ──────────────────────────────────────────────────────────────

def _winsorize_series(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Substitui valores de GMV acima de Q3 + 2.5*IQR pela mediana da
    mesma semana ISO do ano anterior. Se não há referência, usa o cap.
    Retorna série com coluna extra 'y_original' para debug.
    """
    ts = ts.copy().reset_index(drop=True)
    ts["y_original"] = ts["y"].copy()

    q1  = ts["y"].quantile(0.25)
    q3  = ts["y"].quantile(0.75)
    iqr = q3 - q1
    cap = q3 + 2.5 * iqr

    # Calcula ISO week para todas as linhas
    iso_weeks = ts["ds"].dt.isocalendar().week.astype(int)
    years     = ts["ds"].dt.year.astype(int)

    outlier_idx = ts.index[ts["y"] > cap].tolist()
    n_capped = 0

    for idx in outlier_idx:
        iso_w    = int(iso_weeks[idx])
        this_yr  = int(years[idx])
        prev_yr  = this_yr - 1

        # Semanas do ano anterior com o mesmo número ISO, sem serem outliers
        ref_mask = (iso_weeks == iso_w) & (years == prev_yr) & (ts["y"] <= cap)
        ref_vals = ts.loc[ref_mask, "y"]

        if not ref_vals.empty:
            replacement = ref_vals.median()
        else:
            replacement = cap

        ts.at[idx, "y"] = replacement
        n_capped += 1

    if n_capped:
        print(f"    Winsorização: {n_capped} semana(s) acima de Q3+2.5×IQR corrigida(s)"
              f" (cap = R${cap:,.0f})")
    return ts


# ──────────────────────────────────────────────────────────────
# Modelos
# ──────────────────────────────────────────────────────────────

def _try_import_prophet():
    try:
        from prophet import Prophet
        return Prophet
    except ImportError:
        print("  Prophet nao instalado. Usando regressao linear como fallback.")
        return None


def _linear_trend_forecast(ts: pd.DataFrame, periods: int) -> pd.DataFrame:
    ts = ts.copy().reset_index(drop=True)
    x  = np.arange(len(ts))
    y  = ts["y"].values
    slope, intercept = np.polyfit(x, y, 1)

    last_ds  = ts["ds"].max()
    future_ds = [last_ds + pd.Timedelta(weeks=i + 1) for i in range(periods)]
    future_x  = np.arange(len(ts), len(ts) + periods)
    yhat = slope * future_x + intercept

    hist = ts[["ds", "y"]].copy()
    hist["yhat"]       = slope * x + intercept
    hist["yhat_lower"] = hist["yhat"] * 0.85
    hist["yhat_upper"] = hist["yhat"] * 1.15

    fut = pd.DataFrame({
        "ds":         future_ds,
        "y":          np.nan,
        "yhat":       np.maximum(yhat, 0),
        "yhat_lower": np.maximum(yhat * 0.85, 0),
        "yhat_upper": np.maximum(yhat * 1.15, 0),
    })
    return pd.concat([hist, fut], ignore_index=True)


def _prophet_forecast(ts: pd.DataFrame, periods: int, Prophet,
                      holidays: pd.DataFrame | None = None) -> pd.DataFrame:
    kwargs = dict(
        weekly_seasonality   = True,
        yearly_seasonality   = True,
        seasonality_mode     = "multiplicative",
        changepoint_prior_scale = 0.15,   # mais conservador vs sprint_2 (era 0.3)
        interval_width       = 0.80,
    )
    if holidays is not None:
        kwargs["holidays"] = holidays
        kwargs["holidays_prior_scale"] = 15.0   # alto para absorver BF sem vazar

    model = Prophet(**kwargs)
    model.fit(ts[["ds", "y"]])
    future   = model.make_future_dataframe(periods=periods, freq="W")
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result["yhat"]       = result["yhat"].clip(lower=0)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
    result = result.merge(ts[["ds", "y"]], on="ds", how="left")
    return result


# ──────────────────────────────────────────────────────────────
# Gráfico
# ──────────────────────────────────────────────────────────────

def _plot_forecast(prod_id: str, prod_name: str, result: pd.DataFrame,
                   out_path: str, corrected: bool = False):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    hist = result[result["y"].notna()]
    fut  = result[result["y"].isna()]

    ax.plot(hist["ds"], hist["y"], color="#4fc3f7",
            linewidth=1.8, label="GMV real")
    ax.plot(result["ds"], result["yhat"], color="#ffb74d",
            linewidth=1.5, linestyle="--", label="Previsao")
    ax.fill_between(result["ds"], result["yhat_lower"], result["yhat_upper"],
                    alpha=0.25, color="#ffb74d", label="Intervalo 80%")

    if not fut.empty:
        ax.axvline(fut["ds"].min(), color="#ef5350",
                   linewidth=1, linestyle=":", alpha=0.8)

    tag = " [corrigido]" if corrected else ""
    ax.set_title(f"{prod_name}{tag}", color="white", fontsize=13, pad=12)
    ax.set_xlabel("Semana", color="#b0b8c1", fontsize=10)
    ax.set_ylabel("GMV (R$)", color="#b0b8c1", fontsize=10)
    ax.tick_params(colors="#b0b8c1")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3e")
    ax.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)
    ax.grid(axis="y", color="#2a2d3e", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ──────────────────────────────────────────────────────────────
# Utilitário de tendência
# ──────────────────────────────────────────────────────────────

def _determine_trend(actual_avg: float, forecast_avg: float) -> str:
    pct = (forecast_avg - actual_avg) / (actual_avg + 1e-9)
    if pct >  0.05: return "growing"
    if pct < -0.05: return "declining"
    return "stable"


# ──────────────────────────────────────────────────────────────
# Runners
# ──────────────────────────────────────────────────────────────

def _run_core(output_dir: str, file_prefix: str, corrected: bool = False) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    Prophet  = _try_import_prophet()
    holidays = _build_holidays() if corrected else None

    weekly_path = os.path.join(PROCESSED_DIR, "weekly_gmv.csv")
    weekly = pd.read_csv(weekly_path, parse_dates=["ds"])

    products = weekly[["product_id", "product_name"]].drop_duplicates().values.tolist()
    summary_rows = []

    label = "CORRIGIDO" if corrected else "original"
    print(f"Rodando previsoes ({label}) para {len(products)} produtos ...")

    for prod_id, prod_name in products:
        ts = weekly[weekly["product_id"] == prod_id][["ds", "y"]].copy()
        ts = ts.sort_values("ds").reset_index(drop=True)
        train = ts[ts["ds"] <= TRAIN_CUTOFF].copy()

        if len(train) < 8:
            print(f"  [{prod_id}] Poucos dados, pulando.")
            continue

        print(f"  [{prod_id}] {prod_name[:35]}")

        # Winsorização só no modo corrigido
        if corrected:
            train = _winsorize_series(train)

        if Prophet is not None:
            result = _prophet_forecast(train, FORECAST_WEEKS, Prophet, holidays)
        else:
            result = _linear_trend_forecast(train, FORECAST_WEEKS)

        result = result.drop(columns=["y"], errors="ignore")
        result = result.merge(ts[["ds", "y"]], on="ds", how="left")

        actual_avg   = train["y"].tail(12).mean()
        forecast_avg = result[result["y"].isna()]["yhat"].mean()
        if pd.isna(forecast_avg):
            forecast_avg = actual_avg
        pct_change = (forecast_avg - actual_avg) / (actual_avg + 1e-9) * 100
        trend = _determine_trend(actual_avg, forecast_avg)

        out_path = os.path.join(output_dir, f"{file_prefix}{prod_id}.png")
        _plot_forecast(prod_id, prod_name, result, out_path, corrected=corrected)

        icon = {"growing": "(+)", "stable": "(=)", "declining": "(-)"} [trend]
        print(f"    {trend} {icon}  R${actual_avg:,.0f}/sem -> R${forecast_avg:,.0f}/sem"
              f" ({pct_change:+.1f}%)")

        summary_rows.append({
            "product_id":              prod_id,
            "product_name":            prod_name,
            "trend_direction":         trend,
            "avg_weekly_gmv_actual":   round(actual_avg, 2),
            "avg_weekly_gmv_forecast": round(forecast_avg, 2),
            "pct_change":              round(pct_change, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_name   = "forecast_summary_corrigido.csv" if corrected else "forecast_summary.csv"
    csv_path   = os.path.join(output_dir, csv_name)
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Resumo salvo em {csv_path}")
    return summary_df


def run() -> pd.DataFrame:
    """Forecast original (sprint_2)."""
    return _run_core(SPRINT2_DIR, "forecast_", corrected=False)


def run_corrected() -> pd.DataFrame:
    """Forecast com feriados BR + winsorização (sprint_3)."""
    return _run_core(SPRINT3_DIR, "forecast_corrigido_", corrected=True)


if __name__ == "__main__":
    import sys
    if "--corrected" in sys.argv:
        run_corrected()
    else:
        run()
