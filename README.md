# Checkpoint 5 — Random Forest, XGBoost e LightGBM (previsão de churn)

Data Science & Statistical Computing · Prof. Jones Egydio · FIAP 2026

## Integrantes
| Nome | RM |
|---|---|
| Daniel Duarte | 562508 |
| Felippe Nascimento | 562123 |
| Matheus Hideki | 564970 |
| Kawan Oliveira | 562197 |
| Artur Rodrigues | 564309 |

## Links
- GitHub: https://github.com/dnlduarte/checkpoint5-DataScience-fiap
- Streamlit: https://checkpoint5-datascience-fiap-cjkpabyy69pbmf4r3dna95.streamlit.app/

## Problema
Prever se um cliente de telecom vai cancelar o serviço (**churn**, classificação binária) a partir do perfil, dos serviços contratados e da cobrança. Métrica principal: **ROC-AUC** (classes desbalanceadas, ~27% de churn).

- **Base:** IBM Telco Customer Churn — 7.043 clientes × 21 colunas (`data/Telco-Customer-Churn.csv`).
- **Protocolo:** split 80/20 estratificado (`random_state=42`), CV estratificada dentro do treino, mesmos folds e mesma preparação (pipeline) para os 3 modelos. O teste só é usado no Exercício 7.

## Resultados
| Item | Resultado |
|---|---|
| Modelo final | XGBoost + Grid Search (`n_estimators=100`, `learning_rate=0.1`, `max_depth=2`) |
| ROC-AUC CV | 0,8500 ± 0,0116 |
| ROC-AUC teste | 0,8456 |
| F1 / Recall / Precision (teste) | 0,590 / 0,524 / 0,676 |

Baselines apresentam overfitting forte (gap treino–CV de 0,14 a 0,18); o tuning reduz o gap para 0,01–0,04 com AUC de CV ≈ 0,85 nos três algoritmos. Detalhes, gráficos e interpretações estão no notebook.

## Estrutura
```
checkpoint5_projeto/
├── Checkpoint05_RF_XGBoost_LightGBM.ipynb   # análise completa
├── app.py                                   # aplicação Streamlit
├── requirements.txt
├── model/
│   ├── pipeline_final.joblib                # pipeline final (pré-processamento + modelo)
│   ├── metadata.json                        # colunas, opções, métricas
│   └── comparacao_9_configuracoes.csv       # tabela final do Exercício 6
├── data/
│   ├── Telco-Customer-Churn.csv             # base
│   └── casos_consistencia.csv               # 6 casos de teste (notebook e app)
└── tests/
    └── test_paridade.py                     # teste de paridade notebook × Streamlit
```

## Como executar
```bash
cd checkpoint5_projeto
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Checkpoint05_RF_XGBoost_LightGBM.ipynb   # ou Run All
streamlit run app.py
python tests/test_paridade.py   # confirma que o app exibe a mesma probabilidade do notebook
```
