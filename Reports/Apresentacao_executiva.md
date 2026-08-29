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
