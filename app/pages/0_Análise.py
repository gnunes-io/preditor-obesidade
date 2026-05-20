"""
Página 0 – Análise Exploratória dos Dados (EDA)
Distribuições, correlações e principais insights do dataset de obesidade.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.constants import OBESITY_LABELS_SHORT as OBESITY_LABELS, PALETTE, GENERO_PT

st.set_page_config(page_title="Análise | Preditor de Obesidade", page_icon="📊", layout="wide")
st.title("📊 Análise Exploratória dos Dados")
st.caption("Visão geral do dataset utilizado para treinar o modelo preditivo.")

st.markdown("""
<style>
table td { white-space: normal !important; word-wrap: break-word; max-width: 300px; }
</style>
""", unsafe_allow_html=True)

OBESITY_ORDER = list(OBESITY_LABELS.values())
COLOR_MAP     = dict(zip(OBESITY_ORDER, PALETTE))

LABELS_PT = {
    "Age": "Idade", "Height": "Altura (m)", "Weight": "Peso (kg)", "BMI": "IMC",
    "FCVC": "Consumo de Vegetais", "NCP": "Refeições por Dia",
    "CH2O": "Consumo de Água", "FAF": "Atividade Física (dias/sem)", "TUE": "Uso de Tecnologia",
}

SIM_NAO     = {"yes": "Sim", "no": "Não"}
FREQ_PT     = {"no": "Nunca", "Sometimes": "Às vezes",
               "Frequently": "Frequentemente", "Always": "Sempre"}
TRANSP_PT   = {
    "Automobile": "Automóvel", "Motorbike": "Moto", "Bike": "Bicicleta",
    "Public_Transportation": "Transp. Público", "Walking": "A Pé",
}


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    from db.supabase_client import fetch_all
    df = pd.DataFrame(fetch_all("obesity_data"))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "age": "Age", "height": "Height", "weight": "Weight", "gender": "Gender",
        "favc": "FAVC", "fcvc": "FCVC", "ncp": "NCP", "caec": "CAEC",
        "smoke": "SMOKE", "ch2o": "CH2O", "scc": "SCC", "faf": "FAF",
        "tue": "TUE", "calc": "CALC", "mtrans": "MTRANS", "obesity": "Obesity",
    })
    df["Classe"] = df["Obesity"].map(OBESITY_LABELS).fillna(df["Obesity"])
    df["BMI"]    = df["Weight"] / (df["Height"] ** 2)
    df["Gênero"] = df["Gender"].map(GENERO_PT)
    for col, mapa in [("FAVC", SIM_NAO), ("SMOKE", SIM_NAO), ("SCC", SIM_NAO),
                      ("family_history", SIM_NAO), ("CAEC", FREQ_PT),
                      ("CALC", FREQ_PT), ("MTRANS", TRANSP_PT)]:
        df[col + "_PT"] = df[col].map(mapa).fillna(df[col])
    return df


df = load_data()

# ── Dicionário de Variáveis ───────────────────────────────────────────────────
with st.expander("📖 Ver dicionário completo de variáveis. 17 variáveis do dataset original. Height e Weight são exibidas para contexto mas não são utilizadas no modelo preditivo."):
    dicionario = pd.DataFrame([
        ("Gender",         "Gênero",                    "Sexo biológico",                                     "Feminino, Masculino",                               "Sim"),
        ("Age",            "Idade",                     "Idade em anos",                                      "14–61",                                             "Sim"),
        ("Height",         "Altura",                    "Em metros",                                          "1,45–1,98",                                         "Não"),
        ("Weight",         "Peso",                      "Em kg",                                              "39–173",                                            "Não"),
        ("family_history", "Histórico familiar",        "Excesso de peso na família",                         "Sim, Não",                                          "Sim"),
        ("FAVC",           "Alimentos hipercalóricos",  "Consome alimentos muito calóricos frequentemente",   "Sim, Não",                                          "Sim"),
        ("FCVC",           "Consumo de vegetais",       "Frequência de vegetais (1=raramente, 3=sempre)",     "1–3",                                               "Sim"),
        ("NCP",            "Refeições principais",      "Número de refeições principais por dia",             "1–4",                                               "Sim"),
        ("CAEC",           "Lanches entre refeições",   "Come entre as refeições principais",                 "Nunca, Às vezes, Frequentemente, Sempre",           "Sim"),
        ("SMOKE",          "Hábito de fumar",           "Fuma regularmente",                                  "Sim, Não",                                          "Sim"),
        ("CH2O",           "Consumo de água",           "Litros/dia (1=<1L, 2=1-2L, 3=>2L)",                 "1–3",                                               "Sim"),
        ("SCC",            "Monitora calorias",         "Acompanha ingestão calórica diária",                 "Sim, Não",                                          "Sim"),
        ("FAF",            "Atividade física",          "Dias/semana (0=nenhuma, 3=5 ou mais)",               "0–3",                                               "Sim"),
        ("TUE",            "Uso de tecnologia",         "Horas em telas (0=0-2h, 1=3-5h, 2=>5h)",            "0–2",                                               "Sim"),
        ("CALC",           "Consumo de álcool",         "Frequência de consumo",                              "Nunca, Às vezes, Frequentemente, Sempre",           "Sim"),
        ("MTRANS",         "Transporte habitual",       "Principal meio de deslocamento",                     "Automóvel, Moto, Bicicleta, Transp. Público, A Pé", "Sim"),
        ("Obesity",        "Nível de obesidade",        "Classe alvo do modelo",                              "7 classes (Peso Insuficiente a Obesidade III)",      "Alvo"),
    ], columns=["Variável", "Nome completo", "Descrição", "Valores possíveis", "Usado no ML?"])
    st.table(dicionario)

st.divider()

# ── KPIs ─────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total de registros", f"{len(df):,}")
k2.metric("Classes", df["Obesity"].nunique())
k3.metric("Idade média", f"{int(df['Age'].mean())} anos")
k4.metric("IMC médio", f"{df['BMI'].mean():.1f}")

st.divider()

# ── Distribuição de classes ───────────────────────────────────────────────────
st.subheader("Distribuição das Classes de Obesidade")
dist = df["Classe"].value_counts().reindex(OBESITY_ORDER).dropna().reset_index()
dist.columns = ["Classe", "Contagem"]
fig_dist = px.bar(
    dist, x="Classe", y="Contagem", color="Classe",
    color_discrete_map=COLOR_MAP, text_auto=True,
)
fig_dist.update_layout(showlegend=False, xaxis_tickangle=-30)
st.plotly_chart(fig_dist)

st.divider()

# ── Distribuição por gênero ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuição por Gênero")
    gen = df["Gênero"].value_counts().reset_index()
    gen.columns = ["Gênero", "Contagem"]
    fig_gen = px.bar(gen, x="Gênero", y="Contagem", color="Gênero",
                     color_discrete_sequence=["#5DADE2", "#F1948A"], text_auto=True)
    fig_gen.update_layout(showlegend=False)
    st.plotly_chart(fig_gen)

with col2:
    st.subheader("Classe por Gênero")
    gen_class = df.groupby(["Gênero", "Classe"]).size().reset_index(name="Contagem")
    fig_gc = px.bar(
        gen_class, x="Classe", y="Contagem", color="Gênero", barmode="group",
        color_discrete_sequence=["#5DADE2", "#F1948A"],
        category_orders={"Classe": OBESITY_ORDER},
    )
    fig_gc.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_gc)

st.divider()

# ── Distribuições numéricas ───────────────────────────────────────────────────
st.subheader("Distribuição das Variáveis Numéricas por Classe")

num_col = st.selectbox(
    "👉🏼 Selecione a variável:",
    options=list(LABELS_PT.keys()),
    format_func=lambda x: LABELS_PT[x],
)

fig_box = px.box(
    df, x="Classe", y=num_col, color="Classe",
    color_discrete_map=COLOR_MAP,
    category_orders={"Classe": OBESITY_ORDER},
    labels={num_col: LABELS_PT[num_col]},
    points=False,
)
fig_box.update_layout(showlegend=False, xaxis_tickangle=-30)
st.plotly_chart(fig_box)

st.divider()

# ── IMC vs Peso scatter ───────────────────────────────────────────────────────
st.subheader("IMC × Peso por Classe")
fig_sc = px.scatter(
    df, x="Weight", y="BMI", color="Classe",
    color_discrete_map=COLOR_MAP,
    opacity=0.4,
    labels={"Weight": "Peso (kg)", "BMI": "IMC"},
    category_orders={"Classe": OBESITY_ORDER},
)
fig_sc.update_layout(legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig_sc)

st.divider()

# ── Variáveis categóricas ─────────────────────────────────────────────────────
st.subheader("Variáveis Categóricas")

CAT_MAP = {
    "FAVC_PT":           "Consome hipercalóricos",
    "family_history_PT": "Histórico familiar",
    "SMOKE_PT":          "Fumante",
    "SCC_PT":            "Monitora calorias",
    "CAEC_PT":           "Lanches entre refeições",
    "CALC_PT":           "Consumo de álcool",
    "MTRANS_PT":         "Transporte principal",
}
cat_col = st.selectbox("👉🏼 Selecione a variável categórica:", list(CAT_MAP.keys()),
                       format_func=lambda x: CAT_MAP[x])

cat_data = df.groupby([cat_col, "Classe"]).size().reset_index(name="Contagem")
fig_cat = px.bar(
    cat_data, x=cat_col, y="Contagem", color="Classe",
    color_discrete_map=COLOR_MAP, barmode="stack",
    category_orders={"Classe": OBESITY_ORDER},
    labels={cat_col: CAT_MAP[cat_col]},
)
fig_cat.update_layout(xaxis_tickangle=-20, legend=dict(orientation="h", y=-0.3))
st.plotly_chart(fig_cat)

st.divider()

# ── Correlação ───────────────────────────────────────────────────────────────
st.subheader("Correlação entre Variáveis Numéricas")
numeric_cols = ["Age", "Height", "Weight", "BMI", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
corr = df[numeric_cols].corr().round(2)

corr.index   = [LABELS_PT.get(c, c) for c in corr.index]
corr.columns = [LABELS_PT.get(c, c) for c in corr.columns]

fig_corr = px.imshow(
    corr, text_auto=True, color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1, aspect="auto",
)
fig_corr.update_layout(height=500)
st.plotly_chart(fig_corr)
