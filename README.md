# Classificação de textos - SpaCy
## Geral:

O código lê dados de um arquivo CSV contendo informações sobre eventos de risco e seus tipos associados. Ele pré-processa esses dados, lematizando os textos, removendo stopwords e pontuações, e transformando esses textos em representações numéricas usando vetores de palavras. Em seguida, o código divide esses dados em conjuntos de treinamento e teste, treina um modelo de Regressão Logística usando os dados de treinamento vetorizados e avalia o modelo usando os dados de teste.

Finalmente, o código demonstra como usar o modelo treinado para fazer previsões sobre novos textos de eventos de risco. A matriz de confusão e o relatório de classificação são usados para entender o desempenho do modelo. Este código é uma implementação básica de classificação de texto usando técnicas de processamento de linguagem natural e aprendizado de máquina.
