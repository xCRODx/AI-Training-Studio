# Precisão mista: fp16, bfloat16 e AMP

Precisão mista combina formatos numéricos diferentes em uma execução. O objetivo pode ser reduzir o uso de memória ou aproveitar operações aceleradas, mantendo precisão adequada para a tarefa. O ganho depende do hardware, do modelo e do caminho numérico usado.

## 1. O que muda entre os formatos

Um formato de ponto flutuante distribui bits entre sinal, expoente e fração. Mais bits de expoente ampliam a faixa de valores representáveis; mais bits de fração aumentam a precisão relativa.

| Formato | Tamanho | Faixa e precisão, aproximadamente |
|---|---:|---|
| `float32` | 32 bits | Faixa ampla e cerca de 7 dígitos decimais |
| `float16` | 16 bits | Máximo finito 65.504; menor normal cerca de $6.10\times10^{-5}$; possui subnormais menores |
| `bfloat16` | 16 bits | Faixa de expoente próxima à do `float32`, com menos precisão na fração |

`float16` tem mais precisão na fração que `bfloat16`, mas faixa dinâmica menor. `bfloat16` alcança magnitudes próximas às de `float32`, embora arredonde valores com mais intensidade. Os dois podem alterar resultados numéricos; nenhum é automaticamente adequado para toda operação ou modelo.

## 2. AMP e `autocast`

No PyTorch, AMP (*Automatic Mixed Precision*) permite que `autocast` escolha um tipo apropriado para operações compatíveis. O comportamento é definido por operação e dispositivo: nem toda operação é convertida ao mesmo formato, e a forma exata pode variar entre versões e backends.

```python
import torch

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(entrada)
    loss = criterio(logits, alvos)
```

Esse exemplo pressupõe um dispositivo CUDA que ofereça suporte adequado a `bfloat16`. Verifique o suporte no hardware e na versão do PyTorch. Para outros dispositivos, escolha um tipo e um backend compatíveis.

Com autocast, os parâmetros do modelo normalmente permanecem no tipo em que foram criados; certas operações usam temporariamente outro tipo. Isso difere de converter todo o modelo com `model.to(torch.bfloat16)`. Gradientes, buffers e estado do otimizador também afetam o uso de memória, então reduzir a largura dos operandos não significa necessariamente reduzir pela metade a memória total do treino.

## 3. `GradScaler` e fp16

Em fp16, gradientes muito pequenos podem arredondar para zero. Um `GradScaler` dinâmico multiplica a perda antes do backward e ajusta essa escala conforme valores não finitos são observados:

```python
scaler = torch.amp.GradScaler("cuda")

with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(entrada)
    loss = criterio(logits, alvos)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

O scaler ajuda com o intervalo dos gradientes, mas não corrige toda instabilidade numérica nem impede qualquer overflow no forward. Verifique perdas e gradientes. `bfloat16` frequentemente dispensa escalonamento de perda por ter faixa de expoente maior, mas a escolha deve ser validada para o modelo e o backend.

## 4. Exemplo de comparação numérica

O valor $70\,000$ é representável em `float32` e em `bfloat16`, mas excede o maior valor finito de `float16`. Já perto de $1$, `bfloat16` distingue intervalos maiores entre números adjacentes do que `float16`. Isso ilustra o compromisso entre faixa e precisão.

Os erros de arredondamento durante uma rede se acumulam de forma dependente das operações e dos dados. Não se deve presumir que o ruído sempre se cancele ou que a qualidade permaneça idêntica. Compare convergência e métricas relevantes com uma referência em maior precisão.

## 5. Escolher e validar um formato

- Use `float32` como referência quando a estabilidade ou a compatibilidade forem prioritárias.
- Teste `bfloat16` se hardware e backend oferecerem suporte; a faixa ampla pode ajudar em cálculos sensíveis a overflow/underflow.
- Teste `float16` quando houver suporte eficiente e a precisão for suficiente; considere `GradScaler` no treino.
- Meça memória, velocidade, estabilidade da perda e qualidade final, não apenas o tempo de um passo.
- Se os resultados divergirem, localize as operações sensíveis e mantenha maior precisão nelas quando apropriado.

## Resumo

AMP permite usar precisão reduzida em operações selecionadas, mas os ganhos e a estabilidade dependem do hardware e do modelo. `float16` oferece mais precisão relativa que `bfloat16`, enquanto `bfloat16` tem faixa de expoente maior. `GradScaler` pode ajudar no underflow de gradientes fp16, sem substituir validação numérica e de qualidade.
