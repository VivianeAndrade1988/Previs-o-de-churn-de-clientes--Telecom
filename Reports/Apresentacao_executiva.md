# 📉 Prevendo a Fuga de Clientes: Um Case de Retenção Baseado em Dados
### Apresentação Executiva — Projeto de Previsão de Churn (Telecom)

---

## O ponto de partida

Todo mês, uma operadora de telecomunicações via uma parcela dos seus clientes ir embora. Não era
um vazamento pequeno: de cada **4 clientes, 1 cancelava o serviço**. A diretoria sabia que estava
perdendo receita, mas não sabia **quem** estava prestes a sair, nem **por quê**.

A pergunta que motivou este projeto foi simples de fazer e difícil de responder:

> *"Se a gente soubesse, com antecedência, quais clientes estão prestes a cancelar, conseguiríamos
> agir a tempo de reter alguns deles?"*

Este documento conta a jornada de como transformamos **7.043 registros de clientes** em um modelo
preditivo — e, principalmente, no que essa análise revela sobre o comportamento de quem cancela.

---

## Capítulo 1 — O tamanho do problema

Antes de construir qualquer modelo, olhamos para os números crus. De **7.043 clientes** na base,
**1.869 cancelaram o serviço** — uma taxa de churn de **26,5%**.

![Distribuição de clientes por churn](../images/01_churn_distribution.png)


<sub>*Distribuição de clientes: 5.174 permaneceram ativos, 1.869 cancelaram.*</sub>

Para colocar em perspectiva: se nada mudar, a cada 4 clientes que a empresa conquista, ela perde 1
em algum momento. Isso não é um problema de aquisição — é um problema de **retenção**.

E aqui está o primeiro ponto que guiou toda a análise: como essa proporção é desbalanceada (73,5%
ficam vs. 26,5% saem), qualquer modelo "preguiçoso" que simplesmente apostasse "o cliente não vai
cancelar" já acertaria 73% das vezes — e seria completamente inútil na prática. Por isso, medimos
os modelos por métricas mais rigorosas, não só por acurácia.

---

---

## Capítulo 2 — Por que os clientes vão embora?

Com os dados em mãos, começamos a cruzar o perfil de quem cancela contra o de quem fica. Três
padrões se destacaram — e eles contam uma história coerente.

### 🔴 Padrão 1: o contrato mensal é a porta de saída

![Churn por tipo de contrato](../images/02_churn_by_contract.png)

Dos 1.869 clientes que cancelaram, **1.655 (88,6%)** estavam em contratos **mês a mês**. Clientes
com contrato anual ou bianual praticamente não cancelam (166 e 48 casos, respectivamente).

> **Leitura de negócio:** o contrato mensal não tem barreira de saída. É a porta giratória da
> empresa — fácil de entrar, fácil de sair. Migrar clientes para planos com fidelidade é
> provavelmente a alavanca mais direta para reduzir o churn.

### 🔴 Padrão 2: os primeiros meses são os mais perigosos

![Tempo de contrato por churn](../images/07_tenure_hist.png)

O cancelamento se concentra fortemente nos clientes com **pouco tempo de casa**. Clientes que
passam dos primeiros meses tendem a ficar — sugerindo que existe uma "janela crítica" logo no
início da jornada, onde o cliente ainda está decidindo se vale a pena continuar.

> **Leitura de negócio:** o momento de maior risco é o começo do relacionamento. Programas de
> onboarding, acompanhamento proativo nos primeiros 90 dias e incentivos de permanência antecipada
> podem ter um retorno desproporcional.

### 🔴 Padrão 3: a fibra óptica cancela mais que o esperado

![Churn por tipo de contrato](../images/02_churn_by_contract.png)

Clientes com internet **fibra óptica** respondem por **69% dos cancelamentos** (1.297 de 1.869) —
uma proporção bem maior do que seria esperado apenas pela participação desse serviço na base.

> **Leitura de negócio:** isso pode ser sintoma de preço (fibra costuma ser mais cara), de
> qualidade percebida do serviço, ou de um perfil de cliente mais exigente e comparativo (que
> pesquisa concorrentes com mais frequência). Vale investigar a fundo com pesquisa de satisfação
> segmentada por tipo de internet.

### Um sinal que *não* importa: gênero

![Churn por gênero](../images/03_churn_by_gender.png)

Por outro lado, o churn é praticamente idêntico entre homens (930) e mulheres (939). Gênero não é
um fator relevante — é importante saber onde **não** vale a pena direcionar esforço de segmentação.

---

## Capítulo 3 — Construindo a "bola de cristal"

Com os padrões mapeados, o passo seguinte foi treinar modelos capazes de **prever, cliente a
cliente**, a probabilidade de cancelamento. Testamos três abordagens, do mais simples ao mais
sofisticado:

| Modelo | O que é | Papel no projeto |
|---|---|---|
| **Regressão Logística** | Modelo estatístico simples e interpretável | Baseline de comparação |
| **Random Forest** | Centenas de árvores de decisão combinadas | Modelo mais "poderoso", captura padrões complexos |
| **Random Forest (ajustado)** | O mesmo modelo, com parâmetros otimizados | Versão calibrada para evitar erros de excesso de ajuste |

### O resultado (e a surpresa)

![Comparação de métricas entre modelos](../images/09_model_comparison.png)


| Modelo | Acurácia | Identifica corretamente quem cancela (Recall) |
|---|---|---|
| Regressão Logística | 80,1% | 54,2% |
| Random Forest | 78,1% | 46,5% |
| **Random Forest (ajustado)** | 80,0% | 49,0% |

A surpresa foi que o modelo **mais simples (Regressão Logística) teve desempenho equivalente ao
modelo mais sofisticado**. O Random Forest, sem ajustes, "decorou" os exemplos de treino em vez de
aprender um padrão generalizável — um erro comum quando não se toma cuidado na calibração de
modelos complexos.

> **Leitura de negócio:** nem sempre a ferramenta mais avançada é a mais eficaz. O que importa é o
> resultado prático — e aqui, o modelo mais simples se mostrou tão confiável quanto o mais
> complexo, com a vantagem de ser mais fácil de explicar e manter.

### A métrica que realmente importa

Todos os três modelos, hoje, identificam corretamente entre **46% e 54%** dos clientes que de fato
vão cancelar. Em outras palavras: mesmo com o melhor modelo atual, **metade dos cancelamentos
ainda passa despercebido**.

> **Por que isso importa:** para um projeto de retenção, esse é o número mais crítico de todos —
> mais até que a acurácia geral. Cada cliente que o modelo "não vê" é uma oportunidade de retenção
> perdida. Esse é o principal ponto de melhoria para a próxima fase do projeto (ver Próximos Passos).

---

## O que aprendemos, em uma frase

> **O cliente que mais cancela é o que está há pouco tempo na empresa, com contrato mensal e
> internet fibra óptica — e hoje só conseguimos identificar metade deles antes que aconteça.**

---

## 💡 Recomendações de negócio

Com base nos padrões identificados, sugerimos três frentes de ação, da mais rápida para a mais
estrutural:

### 1. Ação imediata — mirar no perfil de maior risco
Priorizar contato proativo (ligação, e-mail, oferta de retenção) para clientes que combinam os
três fatores de risco: **contrato mensal + poucos meses de casa + fibra óptica**. Esse grupo
concentra a maior densidade de cancelamentos e é o alvo mais eficiente para uma campanha de
retenção com recursos limitados.

### 2. Ação estrutural — reduzir a "porta giratória" do contrato mensal
Criar incentivos claros para migração de contratos mensais para planos anuais (desconto
progressivo, benefícios exclusivos, bônus de fidelidade). Como 88,6% do churn vem desse segmento,
mesmo uma pequena taxa de migração pode gerar um impacto desproporcional na taxa geral de
cancelamento.

### 3. Ação de investigação — entender a fibra óptica
Antes de qualquer ação corretiva, vale investigar **por que** a fibra óptica cancela tanto mais:
é preço, é qualidade de rede, é atendimento, é concorrência? Uma pesquisa de satisfação segmentada
com esse público pode transformar uma correlação estatística em uma causa acionável.

---

## 🔭 Próximos passos

| Prioridade | Ação | Por quê |
|---|---|---|
| 🔥 Alta | Melhorar o **recall** do modelo (hoje entre 46–54%) testando balanceamento de classes e ajuste de threshold | É a métrica mais crítica para o objetivo de negócio: cada cancelamento não identificado é uma chance de retenção perdida |
| 🔥 Alta | Rodar uma campanha piloto de retenção com o grupo de maior risco e medir a taxa real de sucesso | Valida se a previsão do modelo se traduz em resultado prático — o verdadeiro teste do projeto |
| 🟡 Média | Testar modelos mais robustos (XGBoost, LightGBM) com balanceamento de classes | Pode melhorar a captura de cancelamentos sem repetir o overfitting visto no Random Forest padrão |
| 🟡 Média | Incorporar métricas de custo de negócio (custo de reter vs. custo de perder um cliente) na avaliação do modelo | Permite decidir o ponto de corte ideal do modelo com base em impacto financeiro, não só estatística |
| 🟢 Contínua | Reavaliar o modelo periodicamente com dados novos | Padrões de churn mudam com o tempo (concorrência, preços, produtos) |

---

## Fechamento

Este projeto não termina com um modelo treinado — ele começa ali. O valor real está em transformar
os padrões identificados em ação: campanhas direcionadas, ajustes de oferta comercial e,
principalmente, um ciclo contínuo de medir, agir e reavaliar.

Os dados já contaram a história de **quem** está saindo e **por quê**. Agora, a decisão de negócio
é: o que a empresa vai fazer com essa informação antes que o próximo cliente cancele?

---

<sub>📎 Documentação técnica completa (metodologia, código, matrizes de confusão e detalhamento de
cada etapa) disponível em `report.html` e `notebooks/churn_prediction.ipynb` no repositório do
projeto.</sub>
