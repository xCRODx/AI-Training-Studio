# Métricas de treinamento

Métricas resumem aspectos diferentes do aprendizado. Nenhuma, isoladamente, comprova que um modelo é útil: é preciso considerar a tarefa, os dados, o conjunto de validação e a evolução ao longo do tempo.

## 1. Perda e validação

A **função de perda** mede o erro que o treinamento tenta reduzir. Em modelos de linguagem, uma escolha comum é a entropia cruzada:

$$
\mathcal{L}_{CE}=-\frac{1}{N}\sum_{i=1}^{N}\log p(y_i\mid x_i).
$$

Uma perda menor indica que o modelo atribuiu maior probabilidade aos alvos avaliados, de acordo com essa função. Não garante que as respostas sejam factualmente corretas, seguras ou adequadas à tarefa.

Meça a perda em dados de treino e validação. Se a de treino cai enquanto a de validação piora, pode haver sobreajuste, vazamento ou uma diferença entre as distribuições. Verifique também exemplos concretos e a divisão dos dados.

Em otimizadores como AdamW, o *weight decay* costuma ser aplicado separadamente da perda da tarefa. Portanto, a perda reportada não inclui necessariamente um termo de regularização; isso depende de como a função de perda e o otimizador foram definidos.

## 2. Perplexidade

Para entropia cruzada medida em nats, a perplexidade é:

$$
\operatorname{PPL}=e^{\mathcal{L}_{CE}}.
$$

Ela transforma a perda em uma escala relacionada ao número efetivo de alternativas atribuídas pelo modelo. Não é uma contagem literal de palavras possíveis. A comparação só é significativa quando tokenizador, corpus, divisão dos dados e convenções de avaliação são compatíveis.

## 3. Acurácia e métricas da tarefa

Em previsão de tokens, a **acurácia top-$k$** mede a fração de exemplos em que o token correto aparece entre as $k$ previsões de maior probabilidade. Top-1 exige que ele seja o mais provável; top-5 permite que esteja entre os cinco primeiros.

Para outras tarefas, use métricas alinhadas ao objetivo, como precisão, recall, F1, erro absoluto médio ou métricas específicas de geração. A acurácia pode esconder problemas de desbalanceamento, e métricas agregadas podem mascarar erros em subgrupos.

## 4. Gradientes e parâmetros

A norma global do gradiente é:

$$
\lVert g\rVert_2=\sqrt{\sum_i g_i^2}.
$$

Ela ajuda a identificar mudanças de escala ou instabilidade, mas valores adequados dependem do modelo, do tamanho do lote, da função de perda e da forma de acumulação. Observe tendências e picos, não um limite universal.

Também podem ser úteis a norma ou magnitude média dos pesos, a taxa de aprendizado e a fração de gradientes nulos. Defina claramente como cada medida é calculada: por tensor, camada ou modelo inteiro; antes ou depois do clipping; e em qual precisão.

## 5. Métricas de quantização

Para pesos ternários, é possível acompanhar a proporção de valores em $-1$, $0$ e $+1$ **após a quantização**. As proporções somam $100\%$. A fração de zeros descreve a esparsidade da representação quantizada, mas não prova que o runtime deixará de executar as operações correspondentes.

Essas medidas são diagnósticos da representação, não métricas universais de qualidade. Não há um equilíbrio fixo que seja saudável para todas as arquiteturas e tarefas; interprete-as junto da perda de validação e do desempenho final.

Outras arquiteturas podem ter diagnósticos específicos, como saturação de ativações ou uso de unidades. Registre a definição e a forma de cálculo junto com o nome da métrica.

## 6. Recursos e velocidade

Uso de memória, utilização de processadores, tempo por passo e exemplos ou tokens processados por segundo ajudam a encontrar gargalos. Interprete-os com o tamanho do lote e da sequência, o número de dispositivos e o período de medição.

Utilização alta não significa automaticamente que o sistema esteja eficiente; utilização baixa pode indicar espera por dados, sincronização ou uma carga pequena demais. Compare execuções equivalentes e inclua qualidade e memória no diagnóstico.

## 7. Leitura conjunta

| Observação | Possíveis interpretações | O que verificar |
|---|---|---|
| Perda de treino cai e validação sobe | Sobreajuste ou diferença de distribuição | Dados, divisão, duração e regularização |
| Perda oscila ou cresce | Instabilidade ou mudança nos dados | Taxa de aprendizado, precisão, gradientes e lotes |
| Perda cai, mas a métrica da tarefa não melhora | Objetivo ou métrica desalinhados | Exemplos, alvos e critérios de avaliação |
| Memória cresce com o contexto | Cache, ativações ou lote maiores | Comprimento de sequência, batch e arquitetura |
| Throughput muda entre execuções | Hardware ou configuração distintos | Aquecimento, dados, paralelismo e tempo medido |

Um valor isolado raramente basta. Compare curvas de treino e validação, observe a estabilidade, avalie exemplos e mantenha constantes as condições de comparação.

## 8. Boas práticas

1. Defina cada métrica e registre sua unidade e método de cálculo.
2. Separe treino, validação e teste; não use o conjunto de teste para ajustar repetidamente o modelo.
3. Compare configurações com os mesmos dados e condições, sempre que possível.
4. Avalie resultados qualitativos e casos difíceis além das médias agregadas.
5. Registre arquitetura, tokenizador, precisão, configuração de treino e versões relevantes.

## Resumo

- Perda mede o objetivo de otimização; validação ajuda a avaliar generalização.
- Perplexidade é uma transformação da entropia cruzada, não uma medida universal de qualidade.
- Métricas de tarefa, gradientes, parâmetros, quantização e recursos descrevem aspectos distintos.
- Interprete tendências e combinações de sinais, sem assumir limites universais.
