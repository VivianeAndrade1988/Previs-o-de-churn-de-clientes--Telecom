# Previsão de Churn de Clientes — Telecom

Projeto de Ciência de Dados que identifica quais clientes de uma operadora de telecomunicações
têm maior probabilidade de cancelar o serviço (churn), entende por que isso acontece, e estima o
impacto de possíveis ações de retenção — do dado bruto até uma lista priorizada de clientes para a
equipe de retenção agir hoje.

## Objetivo

Uma operadora estava perdendo 1 em cada 4 clientes (26,5% de churn) e não conseguia identificar,
com antecedência, quem estava prestes a cancelar. O projeto percorreu quatro perguntas, em ordem:

1. **O que está acontecendo?** → Análise exploratória dos dados
2. **Por que está acontecendo?** → Identificação de padrões de comportamento
3. **Quem vai cancelar a seguir?** → Modelo preditivo + lista de clientes em risco
4. **O que fazer a respeito — e será que funciona?** → Simulação do impacto de ações de retenção

## Base de dados

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) —
7.043 clientes, 21 colunas (perfil, tipo de contrato, serviços contratados, forma de pagamento,
valores cobrados e se o cliente cancelou ou não).

## A jornada do projeto

### 1) Análise exploratória e tratamento dos dados

Metodologia CRISP-DM: entendimento dos dados, identificação de um problema de qualidade
(`TotalCharges` armazenado como texto por causa de valores em branco) e tratamento antes de
qualquer modelagem.

Principais achados:
- A base é desbalanceada: 73,5% dos clientes ficam, 26,5% cancelam.
- 88,6% dos cancelamentos vêm de clientes com contrato mensal (sem fidelidade).
- A taxa de churn cai de 52,9% (clientes com até 6 meses de casa) para 9,5% (clientes com mais de
  4 anos) — os primeiros meses são o momento de maior risco.
- 69% dos cancelamentos vêm de clientes com internet fibra óptica.

Detalhamento completo, com gráficos de cada etapa: [`reports/report.md`](reports/report.md)
(ou a versão em HTML: [`reports/report.html`](reports/report.html))

### 2) Modelagem preditiva

Três modelos foram treinados e comparados: Regressão Logística (baseline), Random Forest, e
Random Forest com tuning de hiperparâmetros (GridSearchCV).

| Modelo | Acurácia (teste) | Recall | Observação |
|---|---|---|---|
| Regressão Logística | 80,1% | 54,2% | Simples e não teve overfitting |
| Random Forest (padrão) | 78,1% | 46,5% | Overfitting severo (99,8% no treino) |
| Random Forest (ajustado) | 80,0% | 49,0% | Overfitting controlado — modelo usado em produção |

Descoberta interessante: o modelo mais simples (Regressão Logística) teve desempenho equivalente
ao mais sofisticado — reforçando que complexidade não é sinônimo de qualidade.

Código completo, já executado: [`notebooks/churn_prediction.ipynb`](notebooks/churn_prediction.ipynb)

### 3) Lista de clientes em risco, hoje

O modelo treinado foi aplicado aos 5.174 clientes atualmente ativos (que ainda não cancelaram),
gerando uma pontuação de risco individual para cada um.

| Faixa de risco | Qtd. de clientes | Ação recomendada |
|---|---|---|
| Alto (≥50%) | 359 | Contato imediato da equipe de retenção |
| Médio (25–50%) | 945 | Monitoramento / campanhas de engajamento |
| Baixo (<25%) | 3.870 | Base saudável |

- Script: [`src/gerar_clientes_propensos_ao_churn.py`](src/gerar_clientes_propensos_ao_churn.py)
- Resultado: [`outputs/clientes_propensos_ao_churn.csv`](outputs/clientes_propensos_ao_churn.csv) / [`.xlsx`](outputs/clientes_propensos_ao_churn.xlsx)
- Gráfico dos 10 clientes mais críticos: [`src/grafico_top10_clientes_risco.py`](src/grafico_top10_clientes_risco.py)

### 4) Simulação: será que uma ação realmente reduz o churn?

O modelo treinado foi usado como motor de simulação "e se": mudamos artificialmente um atributo
de um cliente (ex.: contrato de mensal para anual) e recalculamos a probabilidade de churn,
mantendo tudo o mais igual.

| Ação simulada | Clientes afetados | Taxa de churn da base |
|---|---|---|
| Migrar contrato mensal → anual | 2.220 | 16,75% → 11,71% (−5,04 p.p.) |
| Trocar cheque eletrônico → débito automático | 1.294 | 16,75% → 16,05% (−0,70 p.p.) |
| Desconto de 15% na mensalidade (fibra) | 1.799 | 16,75% → 16,35% (−0,41 p.p.) |
| Ação combinada nos clientes de alto risco | 359 | 16,75% → 15,01% (−1,75 p.p.) |

Importante: isso é uma estimativa baseada em correlações históricas do modelo, não uma prova
causal definitiva — não existe (ainda) um teste A/B real por trás desses números. A forma de
transformar isso em prova é rodar a ação com um grupo piloto e medir o resultado de fato.

Script: [`src/simular_reducao_churn.py`](src/simular_reducao_churn.py)

## Estrutura do repositório

```
previsao-churn-telecom/
├── data/
│   └── Telecom_Churn.csv              # base de dados original
├── notebooks/
│   └── churn_prediction.ipynb         # EDA + modelagem, já executado
├── src/
│   ├── pipeline.py                    # pipeline completo (EDA + treino + avaliação)
│   ├── gerar_clientes_propensos_ao_churn.py
│   ├── grafico_top10_clientes_risco.py
│   └── simular_reducao_churn.py
├── reports/
│   ├── report.md                      # relatório técnico completo (markdown)
│   ├── report.html                    # mesmo relatório, versão HTML
│   └── apresentacao-executiva.md      # storytelling executivo
├── presentations/
│   └── churn_apresentacao_executiva.pptx
├── outputs/                           # resultados já gerados pelos scripts
│   ├── clientes_propensos_ao_churn.csv
│   ├── clientes_propensos_ao_churn.xlsx
│   ├── simulacao_impacto_cenarios.csv
│   └── top10_clientes_risco.png
├── images/                            # gráficos usados nos relatórios
├── requirements.txt
├── LICENSE
└── README.md
```

## Como rodar tudo, na ordem

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Notebook completo (EDA + modelagem)
jupyter notebook notebooks/churn_prediction.ipynb

# 3. Rodar os scripts a partir da pasta src/ (todos esperam data/, images/ e
#    outputs/ um nível acima, na raiz do projeto)
cd src

python pipeline.py                              # roda o pipeline completo (EDA + treino), gera gráficos em ../images
python gerar_clientes_propensos_ao_churn.py     # gera a lista de clientes em risco, salva em ../outputs
python grafico_top10_clientes_risco.py          # gera o gráfico dos 10 mais críticos (depende do anterior)
python simular_reducao_churn.py                 # simula o impacto de ações de retenção
```

## Stack

Python 3 · pandas · numpy · scikit-learn · matplotlib · seaborn · openpyxl · Jupyter Notebook

## Recomendações de negócio

1. **Ação imediata** — priorizar contato de retenção nos 359 clientes de alto risco (lista pronta
   em `outputs/clientes_propensos_ao_churn.csv`).
2. **Ação estrutural** — criar incentivos para migração de contrato mensal para anual: é a
   alavanca de maior impacto estimado (redução de ~5 pontos percentuais na taxa de churn).
3. **Ação investigativa** — entender, com pesquisa de satisfação, por que a fibra óptica concentra
   69% dos cancelamentos (preço? qualidade? concorrência?).

## Próximos passos

- [ ] Rodar uma campanha piloto real com os clientes de alto risco e medir a taxa de sucesso —
  validação definitiva de que a ação funciona (teste A/B).
- [ ] Melhorar o recall do modelo (hoje entre 46–54%) com balanceamento de classes e ajuste de
  threshold.
- [ ] Testar modelos de gradient boosting (XGBoost, LightGBM).
- [ ] Incorporar custo de negócio (custo de reter vs. custo de perder um cliente) na definição do
  ponto de corte do modelo.
- [ ] Reavaliar e retreinar o modelo periodicamente — padrões de churn mudam com o tempo.

## Licença

Distribuído sob os termos definidos em [`LICENSE`](LICENSE).
