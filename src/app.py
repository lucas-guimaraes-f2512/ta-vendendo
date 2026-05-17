"""
Tá vendendo? — Interface Streamlit completa.
Execute com: streamlit run src/app.py  (a partir da raiz do projeto)
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path ──────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from ai.rag import answer, answer_with_debug  # noqa: E402
from ai.vector_store import ensure_indexed    # noqa: E402

# ── Paleta de cores unificada ─────────────────────────────────
PALETTE = {
    "primary":  "#4fc3f7",   # azul claro — cor base do app
    "success":  "#66bb6a",   # verde
    "warning":  "#ffa726",   # laranja
    "danger":   "#ef5350",   # vermelho
    "neutral":  "#90a4ae",   # cinza azulado
    "products": [            # 8 cores para pills e área do GMV
        "#4fc3f7", "#66bb6a", "#ffa726", "#f06292",
        "#ba68c8", "#4db6ac", "#e57373", "#fff176",
    ],
}

# ── Persistência do histórico de chat ─────────────────────────
HISTORY_FILE = Path("/tmp/ta_vendendo_chat_history.json")


def load_history() -> list:
    """Carrega histórico salvo. Retorna lista vazia se não existir."""
    try:
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                return data
    except Exception:
        pass
    return [
        {
            "role": "assistant",
            "content": (
                "Olá, Carlos! Sou seu copiloto de vendas. "
                "Posso analisar tendências, alertar sobre produtos em risco "
                "e sugerir ações práticas. O que você quer saber hoje?"
            ),
        }
    ]


def save_history(messages: list) -> None:
    """Salva o histórico em disco."""
    try:
        HISTORY_FILE.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


@st.cache_resource
def init_vector_store():
    """Garante que o ChromaDB está indexado antes do primeiro uso."""
    already = ensure_indexed()
    return "ready" if already else "re-indexed"


_vs_status = init_vector_store()

# ── Configuração de página ─────────────────────────────────────
st.set_page_config(
    page_title="Tá vendendo?",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (apenas regras em uso) ─────────────────────────────────
st.markdown("""
<style>
    .stProgress > div > div { border-radius: 4px; }
    [data-testid="stSidebar"] { background: #111827; }
    button[data-testid="stBaseButton-secondary"] {
        opacity: 0; position: absolute; top: 0; left: 0;
        width: 100%; height: 100%; cursor: pointer;
    }
    div[data-testid="column"] { position: relative; }
    /* Remove borda azul padrão dos expanders */
    [data-testid="stExpander"] { border: none !important; }
    /* Área do chat com leve separação do topo */
    [data-testid="stChatMessageContainer"] { margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS (com cache)
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_profile() -> dict:
    path = ROOT_DIR / "data" / "synthetic" / "seller_profile.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_weekly_gmv() -> pd.DataFrame:
    path = ROOT_DIR / "data" / "processed" / "weekly_gmv.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["ds"])


@st.cache_data
def load_forecast() -> pd.DataFrame:
    path = ROOT_DIR / "outputs" / "sprint_3" / "forecast_summary_corrigido.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_churn() -> pd.DataFrame:
    path = ROOT_DIR / "outputs" / "sprint_2" / "churn_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_concentration() -> pd.DataFrame:
    path = ROOT_DIR / "outputs" / "sprint_4" / "revenue_concentration.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_seasonality() -> pd.DataFrame:
    path = ROOT_DIR / "outputs" / "sprint_4" / "seasonality_index.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_net_revenue() -> pd.DataFrame:
    path = ROOT_DIR / "outputs" / "sprint_4" / "net_revenue.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ── Carrega uma vez ───────────────────────────────────────────
profile       = load_profile()
gmv_df        = load_weekly_gmv()
forecast      = load_forecast()
churn         = load_churn()
concentration = load_concentration()
seasonality   = load_seasonality()
net_revenue   = load_net_revenue()

# ── Métricas derivadas (usadas na sidebar E nos KPIs) ─────────
if not gmv_df.empty:
    gmv_total    = gmv_df["y"].sum()
    ticket_medio = gmv_total / gmv_df["units"].sum()
else:
    gmv_total    = profile.get("avg_monthly_gmv", 0) * 18 if profile else 0.0
    ticket_medio = 0.0

# ── Aviso global se dados estiverem faltando ──────────────────
_missing = [
    name for name, df in [
        ("weekly_gmv.csv", gmv_df),
        ("forecast_summary_corrigido.csv", forecast),
        ("churn_scores.csv", churn),
    ] if isinstance(df, pd.DataFrame) and df.empty
]
if _missing or not profile:
    st.warning(
        f"⚠️ Arquivos ausentes: {', '.join(_missing) or 'seller_profile.json'}. "
        "Execute `python src/data/pipeline.py` e os scripts de modelo antes de abrir o app.",
        icon="⚠️",
    )


# ══════════════════════════════════════════════════════════════
# SIDEBAR — PERFIL DO VENDEDOR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📦 Tá vendendo?")
    st.markdown("*Copiloto de IA para marketplace*")
    st.divider()

    if profile:
        st.markdown(f"### 👤 {profile.get('name', '—')}")
        st.caption(f"ID: `{profile.get('seller_id', '—')}`")
        st.caption(f"Plataforma: **{profile.get('platform', '—')}**")
        st.caption(f"Categoria: {profile.get('category', '—')}")

        st.divider()

        # Reputação
        st.markdown(f"🏅 **{profile.get('reputation_level', '—')}**")

        # Health score com label contextual
        health = profile.get("health_score", 0)
        st.markdown(f"**Saúde da conta:** {health}/100")
        st.progress(health / 100)
        if health >= 80:
            st.caption("✅ Conta saudável (acima de 80)")
        elif health >= 60:
            st.caption("⚠️ Conta em atenção (entre 60 e 80)")
        else:
            st.caption("🚨 Conta em risco (abaixo de 60)")

        st.divider()

        # Métricas financeiras
        st.metric("GMV — 18 meses",      f"R$ {gmv_total:,.0f}")
        st.metric("Ticket médio/unidade", f"R$ {ticket_medio:.2f}")
        st.metric("Review médio",         f"{profile.get('avg_review_score', 0)} ★")
        st.metric("Taxa de devolução",    f"{profile.get('return_rate_pct', 0)}%")

        # Detalhes secundários em expander
        with st.expander("ℹ️ Detalhes da conta"):
            st.caption(f"Ativo desde {profile.get('start_date', '—')}")
            st.caption(f"Fulfillment: {profile.get('fulfillment_type', '—')}")
            st.caption(f"Entrega média: {profile.get('avg_shipping_days', 0)} dias")
    else:
        st.info("Perfil não encontrado. Execute o pipeline de dados.")


# ══════════════════════════════════════════════════════════════
# CONTEÚDO PRINCIPAL — ABAS
# ══════════════════════════════════════════════════════════════

tab_dash, tab_chat = st.tabs(["📊 Dashboard", "💬 Chat com o copiloto"])


# ──────────────────────────────────────────────────────────────
# ABA 1 — DASHBOARD
# ──────────────────────────────────────────────────────────────

with tab_dash:
    st.markdown("## Visão geral das vendas")

    # ── Bloco de KPIs ─────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        if not gmv_df.empty:
            ultima_semana = gmv_df["ds"].max()
            gmv_semana    = gmv_df[gmv_df["ds"] == ultima_semana]["y"].sum()
            st.metric("GMV esta semana", f"R$ {gmv_semana:,.0f}")
        else:
            st.metric("GMV esta semana", "—")

    with kpi2:
        if not churn.empty:
            n_alertas = int(churn["risk_level"].isin(["High", "Medium"]).sum())
            st.metric("Produtos em alerta", str(n_alertas))
        else:
            st.metric("Produtos em alerta", "—")

    with kpi3:
        if not forecast.empty:
            growing   = int((forecast["trend_direction"] == "growing").sum())
            declining = int((forecast["trend_direction"] == "declining").sum())
            total     = len(forecast)
            if growing > total / 2:
                tendencia_geral = "🟢 Maioria crescendo"
            elif declining > total / 2:
                tendencia_geral = "🔴 Maioria caindo"
            else:
                tendencia_geral = "⚪ Misto"
            st.metric("Tendência geral", tendencia_geral)
        else:
            st.metric("Tendência geral", "—")

    with kpi4:
        st.metric("Ticket médio", f"R$ {ticket_medio:.2f}" if ticket_medio else "—")

    with kpi5:
        if not net_revenue.empty:
            receita_liq_total = net_revenue["receita_liquida"].sum()
            st.metric("Receita líquida estimada", f"R$ {receita_liq_total:,.0f}")
        else:
            st.metric("Receita líquida estimada", "—")

    st.divider()

    # ── Central de alertas ────────────────────────────────────
    st.markdown("### 🔔 Central de alertas")
    _alertas_exibidos = 0

    # 1. Churn — High primeiro, depois Medium
    if not churn.empty:
        for nivel in ["High", "Medium"]:
            for _, row in churn[churn["risk_level"] == nivel].iterrows():
                icon = "🚨" if nivel == "High" else "⚠️"
                st.warning(
                    f"{icon} **{row['product_name']}** — risco {nivel.lower()} "
                    f"(score {row['churn_risk_score']:.2f})  \n"
                    f"_Sinal: {row['main_signal']}_"
                )
                _alertas_exibidos += 1

    # 2. Sazonalidade — mês atual e próximo
    if not seasonality.empty:
        import datetime
        mes_atual = datetime.date.today().month
        mes_prox  = (mes_atual % 12) + 1
        for mes in [mes_atual, mes_prox]:
            row_s = seasonality[seasonality["mes"] == mes]
            if not row_s.empty:
                row_s = row_s.iloc[0]
                if row_s["classificacao"] == "pico":
                    st.success(
                        f"📈 **{row_s['mes_nome']}** é historicamente um mês de pico "
                        f"(índice {row_s['indice_sazonal']:.2f}×). Verifique seu estoque."
                    )
                    _alertas_exibidos += 1
                elif row_s["classificacao"] == "baixo":
                    st.warning(
                        f"📉 **{row_s['mes_nome']}** é historicamente um mês fraco "
                        f"(índice {row_s['indice_sazonal']:.2f}×). Ajuste expectativas."
                    )
                    _alertas_exibidos += 1

    # 3. Concentração crítica de receita
    if not concentration.empty:
        for _, row in concentration[concentration["classificacao"] == "crítico"].iterrows():
            st.error(
                f"⚠️ **{row['product_name']}** representa **{row['pct_receita']:.1f}%** "
                f"do faturamento total. Dependência crítica — monitore de perto."
            )
            _alertas_exibidos += 1

    if _alertas_exibidos == 0:
        st.success("✅ Nenhum alerta no momento. Tudo em ordem.")

    # ── Filtro visual com botões coloridos (fora das colunas) ────
    if not gmv_df.empty:
        PRODUCT_COLORS = {i: c for i, c in enumerate(PALETTE["products"])}

        todos_produtos = sorted(gmv_df["product_name"].unique().tolist())

        if "produtos_ativos" not in st.session_state:
            st.session_state["produtos_ativos"] = set(todos_produtos)

        st.markdown("**Filtrar por produto:**")
        cols_btn = st.columns(len(todos_produtos))
        for i, produto in enumerate(todos_produtos):
            cor = PRODUCT_COLORS.get(i, "#90a4ae")
            ativo = produto in st.session_state["produtos_ativos"]
            nome_curto = " ".join(produto.split()[:2])
            bg = cor if ativo else "transparent"
            borda = f"2px solid {cor}"
            texto_cor = "#111" if ativo else cor
            with cols_btn[i]:
                st.markdown(
                    f"""<div style="
                        background:{bg};border:{borda};border-radius:20px;
                        padding:5px 10px;text-align:center;font-size:11px;
                        font-weight:600;color:{texto_cor};cursor:pointer;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                    ">{nome_curto}</div>""",
                    unsafe_allow_html=True,
                )
                if st.button(" ", key=f"btn_{i}", use_container_width=True,
                             help=produto):
                    if ativo:
                        st.session_state["produtos_ativos"].discard(produto)
                    else:
                        st.session_state["produtos_ativos"].add(produto)
                    st.rerun()

        col_all, col_none = st.columns(2)
        with col_all:
            if st.button("Selecionar todos", use_container_width=True):
                st.session_state["produtos_ativos"] = set(todos_produtos)
                st.rerun()
        with col_none:
            if st.button("Limpar seleção", use_container_width=True):
                st.session_state["produtos_ativos"] = set()
                st.rerun()

        selecionados = list(st.session_state["produtos_ativos"])
        gmv_filtrado = (
            gmv_df[gmv_df["product_name"].isin(selecionados)]
            if selecionados else gmv_df
        )

    st.divider()

    # ── Gráficos lado a lado ───────────────────────────────────
    col_gmv, col_churn = st.columns(2, gap="large")

    with col_gmv:
        st.markdown("### GMV semanal — todos os produtos")
        if not gmv_df.empty:
            gmv_semanal = (
                gmv_filtrado.groupby("ds")["y"]
                .sum()
                .reset_index()
                .rename(columns={"ds": "Semana", "y": "GMV (R$)"})
                .set_index("Semana")
            )
            st.area_chart(gmv_semanal, color=PALETTE["primary"])
        else:
            st.info("Dados de GMV não disponíveis.")

    with col_churn:
        st.markdown("### Risco de saída do catálogo por produto")
        if not churn.empty:
            COLOR_MAP   = {"Low": PALETTE["success"], "Medium": PALETTE["warning"], "High": PALETTE["danger"]}
            churn_sorted = churn.sort_values("churn_risk_score", ascending=True)
            cores = [COLOR_MAP.get(lvl, "#90a4ae") for lvl in churn_sorted["risk_level"]]
            nomes_curtos = [" ".join(n.split()[:3]) for n in churn_sorted["product_name"]]

            fig = go.Figure(go.Bar(
                x=churn_sorted["churn_risk_score"],
                y=nomes_curtos,
                orientation="h",
                marker_color=cores,
                text=[
                    f"{s:.2f} — {lvl}"
                    for s, lvl in zip(
                        churn_sorted["churn_risk_score"],
                        churn_sorted["risk_level"],
                    )
                ],
                textposition="outside",
                cliponaxis=False,
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 1.15], title="Score de risco (0 = seguro · 1 = crítico)"),
                yaxis=dict(title=""),
                margin=dict(l=10, r=10, t=10, b=30),
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟢 Baixo   🟠 Médio   🔴 Alto")
        else:
            st.info("Dados de churn não disponíveis.")

    # ── Tabela de tendências com highlight por cor ─────────────
    st.markdown("### Tendências dos seus produtos (próximas 12 semanas)")

    if not forecast.empty:
        TREND_ICON = {
            "growing":  "🟢 Crescendo",
            "stable":   "⚪ Estável",
            "declining": "🔴 Caindo",
        }
        ROW_BG = {
            "declining": "background-color:rgba(239,83,80,0.15);",
            "growing":   "background-color:rgba(76,175,80,0.10);",
            "stable":    "",
        }

        header = (
            "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
            "<thead><tr style='background:#1e2130;color:#ffffff;'>"
            "<th style='padding:8px 12px;text-align:left;'>Produto</th>"
            "<th style='padding:8px 12px;text-align:center;'>Tendência</th>"
            "<th style='padding:8px 12px;text-align:right;'>GMV atual/sem</th>"
            "<th style='padding:8px 12px;text-align:right;'>GMV previsto/sem</th>"
            "<th style='padding:8px 12px;text-align:right;'>Variação</th>"
            "</tr></thead><tbody>"
        )

        rows_html = ""
        for _, row in forecast.iterrows():
            direction = row["trend_direction"]
            bg        = ROW_BG.get(direction, "")
            icon      = TREND_ICON.get(direction, direction)
            pct       = row["pct_change"]
            pct_color = "#4caf50" if pct > 0 else ("#ef5350" if pct < 0 else "#90a4ae")
            rows_html += (
                f"<tr style='{bg}'>"
                f"<td style='padding:7px 12px;'>{row['product_name']}</td>"
                f"<td style='padding:7px 12px;text-align:center;'>{icon}</td>"
                f"<td style='padding:7px 12px;text-align:right;'>R$ {row['avg_weekly_gmv_actual']:,.2f}</td>"
                f"<td style='padding:7px 12px;text-align:right;'>R$ {row['avg_weekly_gmv_forecast']:,.2f}</td>"
                f"<td style='padding:7px 12px;text-align:right;color:{pct_color};'><b>{pct:+.1f}%</b></td>"
                "</tr>"
            )

        footer = "</tbody></table>"
        st.markdown(header + rows_html + footer, unsafe_allow_html=True)
    else:
        st.info("Tabela de tendências não disponível. Execute o modelo de forecast.")

    st.divider()

    # ── Concentração de receita ────────────────────────────────
    st.markdown("### Concentração de receita por produto")
    if not concentration.empty:
        conc_sorted = concentration.sort_values("pct_receita", ascending=True)
        cores_conc = [
            PALETTE["danger"]  if c == "crítico"  else
            PALETTE["warning"] if c == "relevante" else
            PALETTE["success"]
            for c in conc_sorted["classificacao"]
        ]
        fig_conc = go.Figure(go.Bar(
            x=conc_sorted["pct_receita"],
            y=[" ".join(n.split()[:3]) for n in conc_sorted["product_name"]],
            orientation="h",
            marker_color=cores_conc,
            text=[f"{v:.1f}%" for v in conc_sorted["pct_receita"]],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_conc.update_layout(
            xaxis=dict(
                range=[0, conc_sorted["pct_receita"].max() * 1.25],
                title="% do faturamento total",
            ),
            yaxis=dict(title=""),
            margin=dict(l=10, r=10, t=10, b=30),
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
        )
        st.plotly_chart(fig_conc, use_container_width=True)
        st.caption("🟢 Saudável   🟠 Atenção   🔴 Dependência crítica")
    else:
        st.info("Dados de concentração não disponíveis. Execute `python src/models/business_insights.py`.")


# ──────────────────────────────────────────────────────────────
# ABA 2 — CHAT COM O COPILOTO
# ──────────────────────────────────────────────────────────────

with tab_chat:

    # ── Inicializa histórico ───────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = load_history()

    # ── Botões de ação rápida ──────────────────────────────────
    st.markdown("**Ações rápidas:**")
    btn_cols = st.columns(4, gap="small")
    quick_questions = {
        0: ("📉 Por que as vendas caíram?",       "Por que minhas vendas caíram essa semana?"),
        1: ("💰 Devo mexer no preço do USB-C?",   "Devo baixar o preço do Cabo USB-C hoje?"),
        2: ("⚠️ Produtos em risco de catálogo?",  "Quais produtos estou em risco de perder do catálogo?"),
        3: ("🎯 Como aumentar meu ticket médio?", "O que posso fazer para aumentar meu ticket médio?"),
    }

    question_to_ask: str | None = None
    for idx, (label, question) in quick_questions.items():
        with btn_cols[idx]:
            if st.button(label, use_container_width=True, key=f"quick_{idx}"):
                question_to_ask = question

    with st.expander("⚙️ Opções", expanded=False):
        if st.button("🗑️ Limpar conversa", key="clear_chat"):
            HISTORY_FILE.unlink(missing_ok=True)
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "Olá, Carlos! Conversa reiniciada. "
                        "O que você quer saber sobre suas vendas?"
                    ),
                }
            ]
            st.rerun()

    # ── Histórico de conversa ──────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Campo de texto livre ───────────────────────────────────
    user_input = st.chat_input("Pergunte algo sobre suas vendas...")
    if user_input:
        question_to_ask = user_input

    # ── Processa pergunta ──────────────────────────────────────
    if question_to_ask:
        st.session_state["messages"].append({"role": "user", "content": question_to_ask})
        with st.chat_message("user"):
            st.markdown(question_to_ask)

        with st.chat_message("assistant"):
            with st.spinner("Analisando seus dados..."):
                try:
                    debug    = answer_with_debug(question_to_ask, seller_id="ML-CARLOS-2020")
                    response = debug["answer"]
                    sources  = debug["chunks_used"]
                except Exception as exc:
                    response = (
                        "⚠️ Não consegui consultar o copiloto agora. "
                        "Verifique se a chave OpenAI está configurada no arquivo `.env`.\n\n"
                        f"Detalhe técnico: `{exc}`"
                    )
                    sources = []
            st.markdown(response)
            if sources:
                FONTE_LABEL = {
                    "tendencia":      "📈 Tendências",
                    "churn":          "⚠️ Risco de catálogo",
                    "resumo":         "📋 Resumo executivo",
                    "perfil":         "👤 Perfil",
                    "limitacao":      "ℹ️ Limitações",
                    "concentracao":   "💰 Concentração de receita",
                    "sazonalidade":   "📅 Sazonalidade",
                    "receita_liquida":"💵 Receita líquida",
                }
                tipos_usados = list(dict.fromkeys(c["tipo"] for c in sources))
                st.caption(
                    "Fontes consultadas: "
                    + " · ".join(FONTE_LABEL.get(t, t) for t in tipos_usados)
                )

        st.session_state["messages"].append({"role": "assistant", "content": response})
        save_history(st.session_state["messages"])


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════

col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.caption(
        "**Tá vendendo?** · MVP acadêmico · MBA IBMEC  "
        "· Modelo: gpt-4o-mini + paraphrase-multilingual-MiniLM-L12-v2"
    )
with col_f2:
    st.caption("⚠️ _Dados sintéticos — não refletem vendas reais_")
