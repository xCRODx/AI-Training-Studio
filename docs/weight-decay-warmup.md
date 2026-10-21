# Weight decay, warmup e clipping de gradientes

Taxa de aprendizado, regularização e clipping atuam de maneiras diferentes. Eles podem ajudar a controlar o treinamento, mas nenhum deles garante estabilidade ou generalização sozinho.

## 1. Weight decay e regularização L2

A regularização L2 acrescenta um termo à função de perda:

$$
\mathcal{L}_{total}=\mathcal{L}_{tarefa}+\frac{\lambda}{2}\sum_i w_i^2.
$$

Seu gradiente inclui um termo proporcional aos pesos. O *weight decay* reduz a magnitude dos parâmetros ao longo das atualizações, conforme a regra definida pelo otimizador.

Com SGD simples, L2 e weight decay podem produzir atualizações equivalentes sob a mesma convenção de parâmetros. Em otimizadores adaptativos, essa equivalência geralmente não vale. O AdamW aplica o decay desacoplado da atualização adaptativa:

$$
w \leftarrow w-\eta\frac{\widehat m}{\sqrt{\widehat v}+\epsilon}-\eta\lambda w.
$$

O efeito depende dos parâmetros selecionados. Algumas configurações excluem biases e parâmetros de normalização do decay; a escolha deve ser documentada e avaliada.

## 2. Warmup da taxa de aprendizado

Warmup aumenta gradualmente a taxa de aprendizado no começo do treinamento. Em um warmup linear de $w$ passos:

$$
\eta(t)=\eta_{max}\frac{t}{w},\qquad 0\le t\le w.
$$

Ele pode reduzir atualizações muito grandes no início, especialmente em configurações nas quais o otimizador ou as estatísticas ainda estão se ajustando. Nem todo treino precisa do mesmo warmup; valide sua duração junto com o restante do cronograma.

## 3. Norma e clipping dos gradientes

A norma L2 global é:

$$
\lVert g\rVert_2=\sqrt{\sum_i g_i^2}.
$$

Clipping por norma limita o tamanho do gradiente a um teto $C$:

$$
\widetilde g=g\cdot\min\left(1,\frac{C}{\lVert g\rVert_2}\right).
$$

O clipping global redimensiona todos os componentes pelo mesmo fator e preserva a direção. Ele pode conter picos, mas não corrige dados incorretos, uma taxa inadequada ou uma divergência persistente. A norma deve ser interpretada conforme o modelo e a escala da tarefa, sem um limite universal.

## 4. Exemplo de fluxo em PyTorch

Este fragmento ilustra como aplicar warmup linear e clipping. Defina a função de perda, os parâmetros e o dispositivo de acordo com a tarefa:

```python
import torch

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=peak_lr,
    weight_decay=weight_decay,
)

for step, (inputs, targets) in enumerate(train_loader):
    lr = peak_lr * min(1.0, (step + 1) / warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    loss = loss_function(predictions, targets)
    loss.backward()

    # O valor retornado é a norma anterior ao clipping.
    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm
    )
    optimizer.step()
```

Com acumulação de gradientes, o clipping costuma ser aplicado após acumular os gradientes do passo efetivo. Em precisão mista, pode ser necessário desescalar os gradientes antes de medi-los e limitá-los.

## 5. Interpretar sinais

| Sinal observado | Possíveis explicações | O que verificar |
|---|---|---|
| Norma do gradiente aumenta por vários passos | Instabilidade, lotes atípicos ou escala alterada | Dados, taxa, precisão e função de perda |
| Norma muito pequena ou nula | Gradientes ausentes, saturação ou perda pouco informativa | Grafo, ativações e alvos |
| Perda de treino cai e validação piora | Sobreajuste ou divisão inadequada | Dados, duração e regularização |
| Clipping ocorre em muitos passos | Limite pode estar baixo ou gradientes instáveis | Norma antes do clipping e taxa |
| Parâmetros diminuem sem melhora na validação | Decay excessivo ou inadequado | Grupos de parâmetros e regularização |

Use tendências e exemplos de validação. Um pico isolado não demonstra que o treino divergiu, assim como uma norma estável não prova que o modelo aprendeu a tarefa.

## 6. Escolha de hiperparâmetros

Não há faixas universais de taxa de aprendizado, weight decay, warmup ou limite de clipping. Comece com valores documentados adequados à arquitetura e à escala do problema, altere poucos fatores por vez e compare usando o mesmo conjunto de validação.

O tamanho de lote pode alterar a variância dos gradientes, mas não existe uma regra única para ajustar a taxa em função do lote. Considere também o número de tokens ou exemplos processados e a duração total do treino.

## Erros comuns

1. Tratar clipping como substituto para corrigir uma taxa de aprendizado ou dados inadequados.
2. Confundir a norma antes do clipping com a norma após o clipping.
3. Presumir que L2 e AdamW são equivalentes em todo otimizador.
4. Aplicar decay indiscriminadamente sem considerar grupos de parâmetros.
5. Copiar hiperparâmetros sem validar no modelo e na tarefa em questão.

## Resumo

- Weight decay e L2 podem ter efeitos diferentes em otimizadores adaptativos.
- Warmup ajusta a taxa no início; sua necessidade e duração dependem da configuração.
- Clipping limita o tamanho do gradiente, mas não resolve sozinho uma instabilidade persistente.
- Registre o método e os valores usados e avalie as tendências junto da validação.
