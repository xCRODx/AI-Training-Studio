# Gradient checkpointing: economizando memória no treinamento

Durante o treinamento, a rede precisa calcular a saída e depois voltar pelas camadas para obter os gradientes. Para fazer essa retropropagação, frameworks como PyTorch normalmente guardam ativações intermediárias do *forward*. Em modelos grandes, guardar todas essas ativações pode consumir muita memória.

O **gradient checkpointing**, também chamado de **activation checkpointing**, reduz esse custo: em vez de guardar todas as ativações, mantém alguns pontos de referência (*checkpoints*) e recalcula partes do *forward* quando precisa delas no *backward*.

> **Em resumo:** troca parte da memória por mais computação. O modelo faz trabalho adicional, mas precisa manter menos ativações armazenadas ao mesmo tempo.

## O que muda — e o que não muda

Sem checkpointing, o fluxo simplificado é:

```text
forward:  entrada → camada 1 → camada 2 → camada 3 → saída
          guarda várias ativações intermediárias
backward: usa as ativações guardadas para calcular os gradientes
```

Com checkpointing, algumas ativações intermediárias são descartadas e refeitas mais tarde:

```text
forward:  entrada → camada 1 → [checkpoint] → camada 2 → [checkpoint] → saída
backward: recalcula trechos necessários e então calcula os gradientes
```

Isso não quantiza nem remove os pesos, e tampouco elimina os gradientes ou todo o estado do otimizador. Portanto, não faz qualquer modelo caber em qualquer GPU: ainda é preciso ter memória para os pesos, os gradientes, o otimizador e as operações em andamento. O ganho exato depende da arquitetura e da implementação.

## Exemplo conceitual em PyTorch

O PyTorch permite marcar uma função ou bloco de camadas para ser recalculado durante a retropropagação:

```python
from torch.utils.checkpoint import checkpoint

# Durante o treinamento, a saída é calculada agora e parte das ativações
# será recalculada no backward em vez de permanecer guardada.
saida = checkpoint(bloco, entrada, use_reentrant=False)
```

O checkpointing interessa principalmente ao treinamento. Na inferência comum, não há cálculo de gradientes e essa economia de ativações para o *backward* normalmente não se aplica.

## Diferença para acumulação de gradientes

As duas técnicas podem ser combinadas, mas atacam partes diferentes do problema:

| Técnica | Como reduz o uso de memória | Principal custo |
|---|---|---|
| Acumulação de gradientes | Permite dividir o lote efetivo em micro-lotes menores; a redução do pico vem de processar menos exemplos por micro-lote | Mais passagens antes de cada atualização dos pesos |
| Gradient checkpointing | Descarta algumas ativações intermediárias e as recalcula no *backward* | Mais computação e, em geral, treinamento mais lento |

A acumulação **não reduz o pico de memória por si só** se o tamanho do micro-lote continuar igual. Ela é útil quando se diminui o micro-lote e se quer somar gradientes de várias passagens antes da atualização. Já o checkpointing reduz a quantidade de ativações mantidas para cada passagem.

## Quando considerar

- O treino fica sem memória durante o *forward* ou o *backward*, embora os pesos e o restante do modelo consigam ser carregados.
- É possível aceitar mais tempo de processamento em troca de menor uso de memória.
- A combinação com micro-lotes menores ainda não foi suficiente.

Uma sequência prática é começar por um modelo e um contexto que caibam no equipamento, reduzir o micro-lote se necessário e usar acumulação para ajustar o lote efetivo. Se as ativações continuarem sendo o gargalo, o checkpointing pode ajudar. GPUs com menos memória podem treinar modelos menores e configurações mais enxutas, mas não há uma configuração que sirva igualmente para todo modelo ou dispositivo.

Para entender como dividir um lote em passagens menores, veja também [acumulação de gradientes](./gradient-accumulation.md) e [contexto, batch size e acumulação](./contexto-batch-acumulacao.md).
