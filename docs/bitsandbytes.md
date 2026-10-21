# bitsandbytes

## 1. O que é o bitsandbytes

O **bitsandbytes** é uma biblioteca de **quantização e otimizadores de baixa memória** para PyTorch, voltada para GPUs NVIDIA. Ela fornece kernels CUDA otimizados que permitem treinar e carregar modelos de linguagem usando muito menos VRAM do que o formato `float32`/`bfloat16` tradicional. Seu nome vem do fato de que ela trabalha "nos bits e bytes" — nos níveis mais baixos da representação numérica.

> **Intuição em uma frase:** enquanto a maioria das bibliotecas assume que todo número é um flutuante de 16 ou 32 bits, o bitsandbytes reescreve a conta para que os números circulem em 8 bits ou até 4 bits, trocando um pouco de precisão por uma enorme economia de memória.


## 2. O problema que ela resolve

### 2.1 A conta da memória

Um modelo de linguagem com $P$ parâmetros em `float32` ocupa $4P$ bytes só de pesos. Num treino com Adam, some mais dois estados por parâmetro (momento e variância) em fp32, totalizando $12P$ bytes. Para $P = 7 \times 10^9$ (7 bilhões), isso dá:

$$
12 \times 7 \times 10^9 \approx 84 \text{ GB}
$$

Nenhuma GPU de consumidor tem isso. O bitsandbytes ataca o problema em duas frentes:

1. **Otimizadores de 8 bits**: mantém os momentos do otimizador em `int8`, reduzindo o estado de otimização de $8P$ para $2P$ bytes.
2. **Carregamento de pesos em 4 bits**: converte os próprios pesos para `int4` (formato NF4), reduzindo o footprint do modelo de $4P$ para $\approx 0.5P$ bytes.

### 2.2 A matemática da quantização afim

A quantização inteira mais comum é a **afim**: mapear um intervalo real $[a, b]$ para um intervalo inteiro $[0, 2^n - 1]$ usando uma escala $s$ e um ponto zero $z$:

$$
q = \text{round}\!\left(\frac{x}{s}\right) + z, \qquad s = \frac{b - a}{2^n - 1}
$$

Para desquantizar, inverte-se:

$$
\hat{x} = s \cdot (q - z)
$$

O erro introduzido é, no máximo, metade do passo $s$. Para 8 bits, o passo é $\frac{b-a}{255}$; para 4 bits, $\frac{b-a}{15}$ — bem mais grosseiro, o que exige formatos mais inteligentes (como o NF4, descrito a seguir) para manter a qualidade.

## 3. As três frentes do bitsandbytes

### 3.1 Otimizadores de 8 bits (`bnb.optim.AdamW8bit`)

A ideia mais antiga do bitsandbytes: em vez de armazenar os estados do Adam em fp32, armazená-los em `int8` e desquantizar apenas na hora do passo. Isso reduz o estado do otimizador de 8 bytes por parâmetro (2 estados × 4 bytes) para 2 bytes por parâmetro, mantendo os pesos-mestre em fp32.

```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
```

O truque interno: cada tensor de estado é dividido em blocos de 2048 valores, e cada bloco recebe sua própria escala. Isso é crucial porque, num único tensor, os valores podem variar ordens de magnitude — uma escala global para o tensor inteiro quantizaria mal os valores pequenos.

### 3.2 `LLM.int8()` e a quantização 8-bit com outlier

O `LLM.int8()` observou um fenômeno curioso nos transformers: a maioria das ativações é bem comportada, mas **0.1% dos valores são outliers** de magnitude muito maior. Quantizar tudo em 8 bits destruiria esses outliers — e são justamente eles que carregam informação importante. A solução foi **decompor a matmul**: as colunas com outliers são processadas em fp16, e o resto em int8.

$$
A @ B = A_{\text{outlier}} @ B_{\text{outlier}} + A_{\text{resto}} @ B_{\text{resto}}
$$

O resultado é um cálculo quase tão barato quanto int8 puro, mas com a precisão preservada dos outliers.

### 3.3 NF4 (Normal Float 4-bit) e QLoRA

O **NF4** é um formato de 4 bits que não usa a quantização afim uniforme. Em vez disso, seus 16 valores são escolhidos para **seguir a distribuição dos pesos reais** (que costumam ser aproximadamente gaussianos), alocando mais níveis onde há mais massa de probabilidade. Tecnicamente, os níveis são os quantis de uma normal padrão, normalizados para o intervalo $[-1, 1]$.

O QLoRA combina NF4 com adaptadores de baixa *rank* (LoRA): o modelo base fica congelado em 4 bits, e só os adaptadores (pequenos) são treinados. É por isso que dá para fazer fine-tuning de um modelo de 65B numa única GPU de 48 GB.

## 5. Exemplo numérico progressivo

### 5.1 Quantização afim em 8 bits

Suponha um grupo de pesos no intervalo real $[-0.5, 0.5]$ e alvo `int8` ($n=8$, então $2^8 - 1 = 255$). A escala é:

$$
s = \frac{0.5 - (-0.5)}{255} = \frac{1.0}{255} \approx 0.00392
$$

Um peso $x = 0.25$ vira:

$$
q = \text{round}\!\left(\frac{0.25}{0.00392}\right) = \text{round}(63.78) = 64
$$

Ao desquantizar: $\hat{x} = 64 \times 0.00392 = 0.25088$, um erro de $\approx 0.00088$ (0.35%). Isso é a quantização 8-bit em ação.

### 5.2 A diferença para 4 bits

Com `int4` ($n=4$, $2^4 - 1 = 15$), a escala seria:

$$
s = \frac{1.0}{15} \approx 0.0667
$$

O mesmo $x = 0.25$ viraria $q = \text{round}(0.25/0.0667) = \text{round}(3.75) = 4$, e $\hat{x} = 4 \times 0.0667 = 0.2667$, um erro de 6.7% — muito maior. É por isso que o 4-bit exige formatos não-uniformes (NF4) para não destruir o modelo.

# Uso idêntico ao torch.optim.AdamW:
# loss.backward(); optimizer.step(); optimizer.zero_grad()
target_dtype = x.dtype
w = self.weight.to(device=x.device, dtype=target_dtype)

scale = w.abs().mean().clamp(min=1e-5)          # escala = média de |w|
weight_scaled = w / scale
weight_quantized = torch.round(weight_scaled).clamp(-1, 1)  # {-1, 0, +1}
weight_ste = (weight_quantized - weight_scaled).detach() + weight_scaled
output = F.linear(x, weight_ste) * scale
if self.weight.is_cuda and self.in_features % 4 == 0:
    codes = (quantized.reshape(-1).to(torch.int16) + 1).to(torch.uint8)
    self._inference_packed_weight = (
        codes[0::4] | (codes[1::4] << 2) | (codes[2::4] << 4) | (codes[3::4] << 6)
    ).contiguous()
