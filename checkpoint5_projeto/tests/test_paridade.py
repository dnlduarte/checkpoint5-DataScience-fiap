"""Teste de paridade notebook x Streamlit.

Digita, no app, as entradas de um caso do teste de consistência e confirma que a
probabilidade exibida coincide com a produzida pelo pipeline salvo (mesma que o notebook).
Rodar na raiz do projeto:  python tests/test_paridade.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parent.parent
meta = json.load(open(RAIZ / "model" / "metadata.json", encoding="utf-8"))
casos = pd.read_csv(RAIZ / "data" / "casos_consistencia.csv")
pipeline = joblib.load(RAIZ / "model" / "pipeline_final.joblib")
COLS = meta["colunas_entrada"]


def testar(linha):
    # lado do notebook: pipeline salvo
    entrada = pd.DataFrame([{c: linha[c] for c in COLS}])[COLS]
    esperado = float(pipeline.predict_proba(entrada)[0, 1])

    # lado do app: preenche os campos "como um usuário" e clica em Prever
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60).run()
    assert not app.exception, app.exception
    for c in COLS:
        v = linha[c]
        if c in ("tenure", "SeniorCitizen"):
            v = int(v)
        elif c in meta["colunas_numericas"]:
            v = float(v)
        else:
            v = str(v)
        alvo = [w for w in (list(app.selectbox) + list(app.number_input)) if w.key == f"in_{c}"][0]
        alvo.set_value(v)
    app.run()
    app.button[0].click().run()
    assert not app.exception, app.exception
    exibido = [m for m in app.metric if m.label == "Probabilidade de churn"][0].value
    return esperado, exibido


falhas = 0
for _, linha in casos.iterrows():
    esperado, exibido = testar(linha)
    ok = exibido == f"{esperado * 100:.2f}%"
    falhas += (not ok)
    print(f"{'OK ' if ok else 'ERRO'} | {linha['caso']:50s} | notebook = {esperado:.4f} ({esperado*100:.2f}%) | app = {exibido}")
sys.exit(1 if falhas else 0)
