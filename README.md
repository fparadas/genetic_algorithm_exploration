# Tarefa Experimental 2 - Algoritmos Genéticos

Este relatório descreve a implementação e análise de um algoritmo genético (GA) próprio e sua comparação com a implementação do GA disponível na biblioteca pymoo. O objetivo é implementar o GA, explicar as escolhas dos operadores e parâmetros, e analisar o desempenho usando a função de Rosenbrock em 2D. Serão apresentados gráficos do comportamento da população ao longo de 100 gerações e uma comparação final usando curvas de convergência média de 30 execuções de cada versão do GA.

## Instruções

```ad-quote
Você deverá produzir a codificação de sua própria versão de Algoritmo Genético (GA) e produzir um relatório contendo:

1. O link de seu código no Google Colab
2. Deverá explicar as suas escolhas para os operadores e parâmetros escolhidos.
3. Gráficos que mostrem o comportamento da população ao longo das gerações na otimização da função de Rosenbroack para 2D, por exemplo (Geração: 1, 25, 50 e 100 --- para 100 gerações).
4. Gere a chamada do GA disponível na biblioteca pymoo, informando os operadores e parâmetros utilizados.
5. Compare as duas versões de GA utilizando um gráfico de curvas de convergência média de 30 execuções de cada versão.
6. Aponte ao fim se alguma versão se sobressai a outra, indicando também uma tabela com os valores médio e de desvio padrão final de fitness do indivíduo best na ultima geração. Inclua na tabela o indivíduo que representa o valor de mediana das 30 execuções.
```



---
# References

[1] C. Marcelino. Material de Aula da Semana 5.

[2] J. Blank and K. Deb. pymoo: multi-objective optimization in python. _IEEE Access_, 8():89497–89509, 2020. [doi:10.1109/ACCESS.2020.2990567](https://doi.org/10.1109/ACCESS.2020.2990567).
