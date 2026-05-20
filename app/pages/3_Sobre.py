"""
Página 3 – Sobre o Projeto
Arquitetura, métricas do modelo (confusão, CV, curva de aprendizado),
feature importance, limitações do dataset e contexto do problema.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from app.constants import OBESITY_LABELS_SHORT, PALETTE

ROOT         = Path(__file__).parent.parent.parent
MODEL_PATH   = ROOT / "ml" / "model.pkl"
METRICS_PATH = ROOT / "ml" / "metrics.json"

def fmt_ts(ts: str) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.strftime("%d/%m/%Y às %H:%M")


st.set_page_config(page_title="Sobre | Preditor de Obesidade", page_icon="📖", layout="wide")
st.title("📖 Sobre o Projeto")

# ── Contexto ─────────────────────────────────────────────────────────────────
st.header("Contexto do Problema")
st.markdown("""
A obesidade é um problema de saúde pública global associado a doenças cardiovasculares,
diabetes tipo 2 e outros agravos. O **ObesityPredict** usa Machine Learning para
classificar pacientes em 7 níveis de obesidade com base em hábitos de vida e medidas físicas,
apoiando a triagem clínica e o planejamento de intervenções.

**Dataset:** UC Irvine ML Repository – *Estimation of Obesity Levels Based on Eating Habits
and Physical Condition* (77% dados sintéticos + 23% reais coletados via questionário web).
""")

# ── Arquitetura ──────────────────────────────────────────────────────────────
st.header("Arquitetura do Sistema")
_arch_img = ROOT / "app" / "assets" / "architecture.png"
if _arch_img.exists():
    st.image(str(_arch_img))
else:
    st.warning("Imagem de arquitetura não encontrada.")

# ── Stack Tecnológica ────────────────────────────────────────────────────────
st.header("Stack Tecnológica")
col1, col2, col3, col4 = st.columns(4)
with col1:
    with st.container(border=True):
        st.markdown("**⚙️ Machine Learning**")
        st.caption("Scikit-learn · RandomForest")
with col2:
    with st.container(border=True):
        st.markdown("**🖥️ Front-end**")
        st.caption("Streamlit · Plotly")
with col3:
    with st.container(border=True):
        st.markdown("**🗄️ Banco de Dados**")
        st.caption("Supabase · PostgreSQL")
with col4:
    with st.container(border=True):
        st.markdown("**🚀 Deploy**")
        st.caption("Streamlit Community Cloud · GitHub")

st.divider()

# ── Pipeline de ML ───────────────────────────────────────────────────────────
st.header("Pipeline de Machine Learning")
st.markdown("""
| Etapa | Detalhe | Justificativa |
|-------|---------|---------------|
| **Fonte** | Supabase `obesity_data` (paginado, todos os registros) | Centraliza dados e permite retreinamento incremental |
| **Features excluídas** | `height` e `weight` removidos | Proxies diretos do IMC — incluí-los seria aprender IMC → classe, não hábitos → classe |
| **Feature Engineering** | Interação atividade×hidratação (`faf × ch2o`), lifestyle score (`faf + ch2o - ncp + fcvc`) | Captura sinergias comportamentais não lineares |
| **Encoding** | OneHotEncoder (categóricas) + StandardScaler (numéricas) | StandardScaler mantido para compatibilidade futura com algoritmos sensíveis a escala (SVM, Regressão Logística) |
| **Modelo** | `RandomForestClassifier` (150 árvores, max_depth=15, min_samples_leaf=3, class_weight=balanced) | Robusto a features correlacionadas; class_weight compensa desbalanceamento residual |
| **Validação** | StratifiedKFold (5 folds) + hold-out 20% | Estratificação garante representatividade das 7 classes em cada fold |
| **Artefatos** | `ml/model.pkl`, `ml/label_encoder.pkl`, `ml/metrics.json` | Versionados por timestamp; mantidas as 2 versões mais recentes |
""")

# ── Métricas do modelo ───────────────────────────────────────────────────────
st.header("Métricas do Modelo")

metrics: dict = {}
if not METRICS_PATH.exists():
    st.info("Métricas não encontradas. Execute `python -m ml.train` para gerá-las.")
else:
    metrics = json.loads(METRICS_PATH.read_text())

    # KPIs (6 colunas)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Acurácia", f"{metrics['test_accuracy']:.1%}")
    k2.metric("Macro-F1",           f"{metrics['macro_f1']:.1%}" if metrics.get('macro_f1') else "N/A")
    k3.metric("ROC-AUC",      f"{metrics['roc_auc_macro']:.4f}" if metrics.get('roc_auc_macro') else "N/A")
    k4.metric("Acurácia CV (5-fold)", f"{metrics['cv_mean']:.1%}")
    k5.metric("Desvio Padrão CV",   f"± {metrics['cv_std']:.1%}")
    k6.metric("Amostras de treino", f"{metrics.get('n_samples', '?'):,}")

    # ── Interpretação das métricas ────────────────────────────────────────────
    with st.expander("🎯 Interpretação das Métricas"):
        st.markdown("""
**Acurácia de 83.5%** significa que, em 100 pacientes do conjunto de teste, o modelo classificou
corretamente 83. Esse resultado foi obtido sem as variáveis de altura e peso: o modelo classifica
obesidade exclusivamente por hábitos de vida, o que é clinicamente mais relevante para triagem
comportamental. Com as biométricas, a acurácia era 93.6% — a queda de 10pp representa a
dificuldade real do problema.

---

**Por que Macro-F1 importa mais em contexto médico?**
A acurácia trata todos os erros igualmente. Mas classificar um paciente com *Obesidade Tipo III*
como *Sobrepeso* é clinicamente muito mais grave do que o inverso. O Macro-F1 calcula a média
do F1-Score por classe *sem ponderação pelo tamanho*, penalizando igualmente erros em classes
raras. É a métrica primária para sistemas de triagem clínica.

---

**O que é ROC-AUC e por que 0.97 aqui?**
O ROC-AUC de 0.9721 indica que, mesmo sem altura e peso, o modelo consegue separar muito bem
as 7 classes de obesidade. Para um problema multiclasse de 7 categorias sem variáveis biométricas
diretas, esse valor é considerado excelente. A linha diagonal tracejada no gráfico representa
um classificador aleatório (AUC=0.5): quanto mais a curva se afasta dela em direção ao canto
superior esquerdo, melhor o modelo.

---

**Limitação dos dados sintéticos**
O modelo foi treinado com 2.111 amostras, das quais apenas 498 são registros reais de pacientes
(México, Peru e Colômbia). Os 1.613 restantes foram gerados artificialmente pelo SMOTE para
equilibrar as classes. Isso garante boas métricas de treinamento, mas *não valida o modelo
para uso clínico real* sem validação adicional com dados hospitalares brasileiros.
        """)


    # ── Por que removemos height e weight? ───────────────────────────────────
    st.markdown("""
**Por que removemos altura e peso do modelo?**

Altura e peso são proxies diretos do IMC, que por sua vez é o critério matemático de classificação
das 7 classes de obesidade. Um modelo que usa essas variáveis aprende essencialmente IMC → classe,
não hábitos → classe. Isso gera acurácia alta mas utilidade clínica baixa: o médico já sabe o peso
do paciente, não precisa de ML para calcular o IMC.

O modelo atual classifica risco de obesidade exclusivamente por comportamentos modificáveis:
alimentação, atividade física, histórico familiar e estilo de vida. Com 83.5% de acurácia e ROC-AUC
0.97 sem as biométricas, o sistema tem valor real de triagem comportamental.
""")

    # Comparação com treino anterior (lê metrics_history.jsonl)
    history_path = METRICS_PATH.parent / "metrics_history.jsonl"
    if history_path.exists():
        runs = [json.loads(l) for l in history_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(runs) >= 2:
            prev = runs[-2]
            delta_acc = metrics["test_accuracy"] - prev["test_accuracy"]
            delta_f1  = metrics.get("macro_f1", 0) - prev.get("macro_f1", 0)
            delta_cv  = metrics["cv_mean"] - prev["cv_mean"]
            delta_std = metrics["cv_std"] - prev["cv_std"]
            with st.expander("📊 Comparação com treino anterior"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Acurácia hold-out", f"{metrics['test_accuracy']:.1%}",
                          delta=f"{delta_acc:+.1%}", delta_color="normal")
                c2.metric("Macro-F1", f"{metrics.get('macro_f1', 0):.1%}",
                          delta=f"{delta_f1:+.1%}", delta_color="normal")
                c3.metric("Acurácia CV média", f"{metrics['cv_mean']:.1%}",
                          delta=f"{delta_cv:+.1%}", delta_color="normal")
                c4.metric("Desvio Padrão CV", f"± {metrics['cv_std']:.1%}",
                          delta=f"{delta_std:+.1%}", delta_color="inverse")
                prev_ts = fmt_ts(prev["trained_at"]) if prev.get("trained_at") else "?"
                curr_ts = fmt_ts(metrics["trained_at"]) if metrics.get("trained_at") else "?"
                st.caption(f"Treino anterior: `{prev_ts}` · Treino atual: `{curr_ts}`")

    # Metadados do treino
    if metrics.get("trained_at"):
        st.caption(
            f"Treinado em: `{fmt_ts(metrics['trained_at'])}` · "
            f"{metrics.get('n_samples', '?')} amostras · "
            f"scikit-learn {metrics.get('sklearn_version', '?')}"
        )

    # CV scores por fold
    st.subheader("Acurácia por Fold (Validação Cruzada)")
    cv_scores = metrics.get("cv_scores", [])
    if cv_scores:
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
            "Acurácia": cv_scores,
        })
        range_min = max(0.0, min(cv_scores) - 0.05)
        fig_cv = px.bar(cv_df, x="Fold", y="Acurácia", text_auto=".1%",
                        color="Acurácia", color_continuous_scale="Blues",
                        range_y=[range_min, 1.0])
        fig_cv.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig_cv)
    else:
        st.warning("CV scores não disponíveis no modelo atual.")

    # Curvas ROC por classe
    roc_data = metrics.get("roc_auc_per_class")
    if roc_data:
        st.subheader("Curvas ROC por Classe (One-vs-Rest)")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray", width=1))
        colors = PALETTE
        classes_roc = list(roc_data.keys())
        for i, cls in enumerate(classes_roc):
            d = roc_data[cls]
            label_pt = OBESITY_LABELS_SHORT.get(cls, cls)
            fig_roc.add_trace(go.Scatter(
                x=d["fpr"], y=d["tpr"],
                mode="lines",
                name=f"{label_pt} (AUC={d['auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig_roc.update_layout(
            xaxis_title="Taxa de Falsos Positivos",
            yaxis_title="Taxa de Verdadeiros Positivos",
            legend=dict(orientation="v", x=1.01, y=1),
            height=500,
        )
        st.plotly_chart(fig_roc)

    # Matriz de confusão
    st.subheader("Matriz de Confusão (Hold-out)")
    cm      = np.array(metrics["confusion_matrix"])
    classes = metrics["class_names"]
    labels_pt = [OBESITY_LABELS_SHORT.get(c, c) for c in classes]
    fig_cm = ff.create_annotated_heatmap(
        z=cm, x=labels_pt, y=labels_pt,
        colorscale="Blues", showscale=True,
    )
    fig_cm.update_layout(
        xaxis_title="Predito", yaxis_title="Real",
        xaxis={"side": "bottom"}, height=500,
    )
    fig_cm["data"][0]["xgap"] = 2
    fig_cm["data"][0]["ygap"] = 2
    st.plotly_chart(fig_cm)

    # Análise de erros
    st.subheader("Análise de Erros")
    st.info(
        "As fronteiras mais difíceis para o modelo são entre Peso Normal e Sobrepeso I, "
        "e entre Sobrepeso II e Obesidade I. Essas classes são clinicamente adjacentes e, "
        "sem as medidas biométricas de altura e peso, o modelo depende exclusivamente de "
        "hábitos para diferenciá-las. Erros nessas fronteiras têm menor impacto clínico "
        "do que confusões entre classes distantes como Peso Insuficiente e Obesidade III."
    )

    # Classification report
    st.subheader("Relatório por Classe")
    report = metrics["classification_report"]
    rows = []
    for cls in classes:
        if cls in report:
            r = report[cls]
            rows.append({
                "Classe": labels_pt[classes.index(cls)],
                "Precisão": f"{r['precision']:.1%}",
                "Recall": f"{r['recall']:.1%}",
                "F1-Score": f"{r['f1-score']:.1%}",
                "Suporte": int(r["support"]),
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

# ── Feature importance ───────────────────────────────────────────────────────
st.header("Feature Importance")

fi_data = metrics.get("feature_importances")
if fi_data:
    fi_df = pd.DataFrame(fi_data).head(20)
    fi_df.columns = ["Feature", "Importância"]
    fig_fi = px.bar(
        fi_df, x="Importância", y="Feature", orientation="h",
        color="Importância", color_continuous_scale="Blues",
        title="Top 20 Features por Importância (Gini)",
    )
    fig_fi.update_layout(yaxis={"autorange": "reversed"}, height=550)
    st.plotly_chart(fig_fi)
    st.caption(
        "Com a remoção de height e weight, age e family_history emergem como os principais "
        "preditores. Isso é consistente com a literatura médica: envelhecimento e predisposição "
        "genética são fatores de risco consolidados para obesidade. "
        "Hábitos como FAVC (consumo de hipercalóricos) e FAF (atividade física) aparecem em "
        "seguida, confirmando que o modelo aprendeu padrões comportamentais relevantes."
    )
else:
    st.info("Feature importance não disponível. Execute `python -m ml.train` para gerá-la.")

# ── Limitações e Próximos Passos ─────────────────────────────────────────────
st.header("⚠️ Limitações e Próximos Passos")

st.warning("""
**Limitações do modelo e do dataset**

**1. 77% dos dados são sintéticos (SMOTE)**
O dataset original contém apenas 498 registros reais. Os demais 1.613 registros foram gerados
artificialmente via SMOTE (*Synthetic Minority Over-sampling Technique*) para balancear as classes.
Dados sintéticos seguem distribuições controladas e matematicamente coerentes, e ajuda muito o modelo
atingir ~85% de acurácia. Esse número **não reflete o desempenho esperado em dados hospitalares reais**,
onde ruído, valores ausentes e variabilidade humana são muito maiores.

**2. Modelo treinado em população específica**
Os dados reais foram coletados via questionário web com participantes do México, Peru e Colômbia,
majoritariamente jovens adultos (14–61 anos). O modelo pode apresentar viés ao ser aplicado em
populações com perfis distintos (idosos, outras etnias, outras regiões geográficas).

**3. Ausência de variáveis clínicas laboratoriais**
O modelo se baseia exclusivamente em dados autodeclarados (hábitos alimentares, estilo de vida,
medidas físicas). Variáveis clínicas fundamentais para o diagnóstico de obesidade (glicemia
de jejum, pressão arterial, perfil lipídico e circunferência abdominal) não estão disponíveis
no dataset e, portanto, não foram consideradas.

**4. Sistema de apoio à decisão, não substituto médico**
Este sistema foi desenvolvido como ferramenta de **triagem inicial e apoio clínico**. Qualquer
resultado deve ser interpretado por um profissional de saúde qualificado com acesso ao histórico
completo do paciente. O modelo não realiza diagnóstico clínico.
""")

st.info("""
**Próximos passos sugeridos para evolução do sistema**

- 🏥 **Retreinamento com dados hospitalares reais:** coletar dados anonimizados de prontuários
  para substituir progressivamente os registros sintéticos e validar o modelo em contexto clínico real

- 🩺 **Adição de variáveis clínicas laboratoriais:** incorporar glicemia, pressão arterial,
  perfil lipídico e circunferência abdominal para aumentar o poder preditivo e a relevância médica

- 👨‍⚕️ **Validação com especialistas médicos:** submeter o modelo a revisão por endocrinologistas
  e nutricionistas antes de qualquer uso clínico, seguindo as diretrizes de IA em saúde (CFM, ANVISA)
""")

