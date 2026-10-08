"""App Streamlit — previsão de churn de clientes (Checkpoint 5).

Usa EXATAMENTE o pipeline salvo pelo notebook (model/pipeline_final.joblib):
o pré-processamento e o modelo final são o mesmo objeto avaliado no teste.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="Previsão de churn", page_icon="📉", layout="wide")


@st.cache_resource
def carregar_artefatos():
    pipeline = joblib.load(BASE / "model" / "pipeline_final.joblib")
    with open(BASE / "model" / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    casos = pd.read_csv(BASE / "data" / "casos_consistencia.csv")
    return pipeline, meta, casos


pipeline, meta, casos = carregar_artefatos()
COLS = meta["colunas_entrada"]
NUM = meta["colunas_numericas"]
OPCOES = meta["opcoes_categoricas"]
FAIXAS = meta["faixas_numericas"]
LIMIAR = meta["limiar"]
SIM_NAO = {0: "Não", 1: "Sim"}


def valor_padrao(col):
    return FAIXAS[col]["mediana"] if col in NUM else OPCOES[col][0]


# valores iniciais dos campos (só na primeira execução)
for c in COLS:
    if f"in_{c}" not in st.session_state:
        v = valor_padrao(c)
        st.session_state[f"in_{c}"] = int(v) if c in ("tenure", "SeniorCitizen") else v


def carregar_caso():
    """Preenche os campos com um caso do teste de consistência do notebook."""
    escolha = st.session_state["caso_escolhido"]
    if escolha == "— preencher manualmente —":
        return
    linha = casos[casos["caso"] == escolha].iloc[0]
    for c in COLS:
        v = linha[c]
        st.session_state[f"in_{c}"] = int(v) if c in ("tenure", "SeniorCitizen") else (float(v) if c in NUM else str(v))


st.title("📉 Previsão de cancelamento de clientes (churn)")
st.caption(f"Modelo final: **{meta['modelo_final']}** · mesmo pipeline avaliado no notebook do Checkpoint 5")

aba_prever, aba_modelo = st.tabs(["Fazer previsão", "Sobre o modelo"])

with aba_prever:
    st.selectbox("Carregar um caso do teste de consistência (opcional)",
                 ["— preencher manualmente —"] + casos["caso"].tolist(),
                 key="caso_escolhido", on_change=carregar_caso)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Perfil")
        st.selectbox("gender", OPCOES["gender"], key="in_gender")
        st.selectbox("SeniorCitizen", [0, 1], format_func=SIM_NAO.get, key="in_SeniorCitizen")
        st.selectbox("Partner", OPCOES["Partner"], key="in_Partner")
        st.selectbox("Dependents", OPCOES["Dependents"], key="in_Dependents")
    with c2:
        st.subheader("Contrato e cobrança")
        st.selectbox("Contract", OPCOES["Contract"], key="in_Contract")
        st.number_input("tenure (meses como cliente)", min_value=0, max_value=100, step=1, key="in_tenure")
        st.number_input("MonthlyCharges (US$/mês)", min_value=0.0, max_value=300.0, step=0.05, key="in_MonthlyCharges")
        st.number_input("TotalCharges (US$ acumulado)", min_value=0.0, max_value=20000.0, step=0.05, key="in_TotalCharges")
        st.selectbox("PaperlessBilling", OPCOES["PaperlessBilling"], key="in_PaperlessBilling")
        st.selectbox("PaymentMethod", OPCOES["PaymentMethod"], key="in_PaymentMethod")
    with c3:
        st.subheader("Serviços")
        for c in ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            st.selectbox(c, OPCOES[c], key=f"in_{c}")

    if st.button("Prever", type="primary"):
        entrada = pd.DataFrame([{c: st.session_state[f"in_{c}"] for c in COLS}])[COLS]
        proba = float(pipeline.predict_proba(entrada)[0, 1])
        classe = int(pipeline.predict(entrada)[0])

        st.divider()
        r1, r2 = st.columns(2)
        r1.metric("Probabilidade de churn", f"{proba * 100:.2f}%")
        r2.metric("Classe prevista", "Vai cancelar (1)" if classe == 1 else "Deve permanecer (0)")
        st.progress(min(max(proba, 0.0), 1.0))
        if classe == 1:
            st.error(f"Risco alto de cancelamento (probabilidade ≥ {LIMIAR:.0%}). Priorizar ação de retenção.")
        else:
            st.success(f"Risco abaixo do limiar de {LIMIAR:.0%}.")
        st.caption(f"Valor exato da probabilidade: {proba:.4f}")
        with st.expander("Dados enviados ao modelo"):
            st.dataframe(entrada.astype(str).T.rename(columns={0: "valor"}))

with aba_modelo:
    st.subheader(meta["modelo_final"])
    st.json(meta["hiperparametros"])
    m = meta["metricas_teste"]
    st.markdown(f"**Validação cruzada (treino):** ROC-AUC = {meta['auc_cv']:.4f} ± {meta['auc_cv_std']:.4f}")
    cols = st.columns(len(m))
    for col, (nome, valor) in zip(cols, m.items()):
        col.metric(f"{nome} (teste)", f"{valor:.3f}")
    st.caption("O limiar de decisão é 0,5. Ferramenta de apoio à decisão: não substitui a análise do time de retenção.")
