# LoRA e PEFT

## 1. A intuição: em vez de reformar a casa, troque uma maçaneta

Considere uma casa já construída e funcional. Você quer que ela sirva para um novo propósito — digamos, receber mais luz natural. Você poderia **reformar a casa inteira**: derrubar paredes, refazer a fundação, trocar toda a estrutura. Caro, arriscado e demorado. Ou você poderia simplesmente **trocar as janelas** — uma mudança pequena e localizada que atinge o objetivo com uma fração do custo.

A **PEFT** (Parameter-Efficient Fine-Tuning, ou *ajuste fino eficiente em parâmetros*) segue exatamente essa lógica. Em vez de atualizar os **bilhões** de parâmetros de um modelo durante o fine-tuning, ela:

1. **Congela** todos os pesos originais do modelo.
2. **Adiciona** um número minúsculo de parâmetros novos.
3. **Treina apenas** esses parâmetros novos.

A **LoRA** (Low-Rank Adaptation, ou *adaptação de baixa ordem*) é a técnica de PEFT mais popular. A sua ideia central é uma observação matemática elegante: **a mudança de pesos necessária para adaptar um modelo a uma nova tarefa pode ser representada por matrizes de baixa ordem** — ou seja, a atualização vive em um subespaço de dimensão muito menor do que a matriz original.

> PEFT adapta um modelo treinando pouquíssimos parâmetros novos, mantendo os pesos originais congelados. LoRA implementa isso representando a atualização de cada matriz de pesos como o produto de duas matrizes pequenas, de baixa ordem.

### A economia em números

Um modelo com $d = 4096$ e $k = 4096$ (dimensões de entrada e saída de uma camada linear) tem:

$$
d \times k = 4096 \times 4096 = 16\,777\,216 \text{ parâmetros}
$$

Se aplicarmos LoRA com posto $r = 16$, os novos parâmetros por camada são:

$$
d \times r + r \times k = 4096 \times 16 + 16 \times 4096 = 131\,072
$$

A redução é de:

$$
\frac{16\,777\,216}{131\,072} = 128\times
$$

Ou seja, treinamos **128 vezes menos** parâmetros nessa camada. E como o posto $r$ costuma ser $8$, $16$ ou $64$, a economia se repete em todas as camadas.

## 2. O conceito formal

### 2.1 A decomposição low-rank

O ponto de partida é uma camada linear com matriz de pesos $W_0 \in \mathbb{R}^{d \times k}$. Em um fine-tuning completo, atualizaríamos $W_0$ diretamente, gerando uma nova matriz $W = W_0 + \Delta W$, onde $\Delta W$ é uma matriz densa de mesmo tamanho.

A LoRA **congela** $W_0$ e **fatora** a atualização:

$$
\Delta W = B A
$$

com:

$$
B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)
$$

A matriz efetiva fica:

$$
W_{\text{efetiva}} = W_0 + \Delta W = W_0 + B A
$$

E a passagem direta (forward) da camada, para uma entrada $h \in \mathbb{R}^{k}$, é:

$$
h' = W_0 h + B A h = W_0 h + B (A h)
$$

Note a beleza: o termo original $W_0 h$ é calculado normalmente (com pesos congelados), e o termo novo $B(Ah)$ é o **adaptador**, calculado em dois passos de dimensão reduzida.

### 2.2 Por que low-rank funciona?

A intuição vem de uma hipótese (hoje amplamente aceita empiricamente): embora os modelos tenham pesos em espaços de dimensão altíssima, **a adaptação a uma tarefa específica vive em um subespaço de baixa dimensão intrínseca**. Em outras palavras, para "empurrar" o modelo de um comportamento genérico para um comportamento específico, você não precisa mexer em todas as direções do espaço de pesos — apenas em algumas poucas direções "relevantes".

Formalmente, dizemos que a atualização $\Delta W$ tem **posto** $r$ baixo, o que significa que ela pode ser escrita como soma de $r$ produtos externos:

$$
\Delta W = \sum_{i=1}^{r} \sigma_i \, u_i v_i^\top
$$

onde $u_i$ e $v_i$ são vetores singulares e $\sigma_i$ os valores singulares. A LoRA simplesmente **aprende diretamente** nessa forma fatorada $B A$, restringindo a atualização ao subespaço de dimensão $r$.

### 2.3 Inicialização e escala

A inicialização é crucial para que o adaptador comece "neutro" (não perturbe o modelo no início do treino):

- $A$ é inicializada com valores **aleatórios gaussianos** $\mathcal{N}(0, \sigma^2)$.
- $B$ é inicializada com **zeros**.

Com isso, no início $\Delta W = B A = 0$, e o modelo se comporta exatamente como o original. O parâmetro $\alpha$ (escala) controla a magnitude da contribuição do adaptador:

$$
h' = W_0 h + \frac{\alpha}{r} \, B A h
$$

A divisão por $r$ evita que a escala cresça com o posto, e $\alpha$ funciona como uma "taxa de aprendizado" da contribuição LoRA.

### 2.4 Exemplo numérico progressivo

Vamos acompanhar numericamente o que acontece em uma camada pequena, com $d = 4$, $k = 4$ e $r = 2$.

Matriz congelada $W_0$:

$$
W_0 = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{pmatrix}
$$

Matrizes do adaptador (inicializadas, antes do treino):

$$
A = \begin{pmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.5 & 0.6 & 0.7 & 0.8 \end{pmatrix}, \quad
B = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}
$$

Como $B = 0$, temos $\Delta W = B A = 0$, e $W_{\text{efetiva}} = W_0$. O modelo começa **idêntico** ao original.

Após algum treino, suponha que $B$ tenha aprendido valores:

$$
B = \begin{pmatrix} 0.05 & -0.02 \\ 0.01 & 0.03 \\ -0.04 & 0.02 \\ 0.02 & 0.01 \end{pmatrix}
$$

Então a atualização $\Delta W = B A$ é:

$$
\Delta W = \begin{pmatrix} 0.05 & -0.02 \\ 0.01 & 0.03 \\ -0.04 & 0.02 \\ 0.02 & 0.01 \end{pmatrix}
\begin{pmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.5 & 0.6 & 0.7 & 0.8 \end{pmatrix}
$$

Calculando a primeira entrada:

$$
\Delta W_{1,1} = 0.05 \cdot 0.1 + (-0.02) \cdot 0.5 = 0.005 - 0.010 = -0.005
$$

E assim por diante. A atualização $\Delta W$ é **pequena** (valores próximos de zero) e **estruturada** (posto 2), mas já empurra o comportamento na direção correta. O modelo original é preservado, e só a "janela" é trocada.

## 3. Por que funciona e quando usar

### 3.1 Vantagens da LoRA/PEFT

1. **Memória drasticamente menor**: não precisamos armazenar gradientes nem estados do otimizador para os bilhões de parâmetros congelados — apenas para os poucos parâmetros do adaptador.
2. **Treino mais rápido**: menos parâmetros, menos computação de retropropagação.
3. **Checkpoints minúsculos**: em vez de salvar o modelo inteiro, salvamos apenas os adaptadores (alguns MB contra GB).
4. **Sem esquecimento catastrófico**: como os pesos originais ficam congelados, o conhecimento geral é preservado por construção.
5. **Troca rápida de tarefas**: você pode ter vários adaptadores (um por tarefa) e trocá-los sem re-treinar o modelo base.

### 3.2 Quando usar LoRA

| Cenário | Recomendação |
|---|---|
| Fine-tuning com poucos recursos (1 GPU, memória limitada) | LoRA é ideal |
| Muitas tarefas/domínios distintos | Um adaptador por tarefa |
| Preservar o conhecimento geral | LoRA (pesos congelados) |
| Mudança drástica de domínio, dados muito grandes | Considere full fine-tuning |
| Modelo muito pequeno (milhões de parâmetros) | Full fine-tuning pode ser viável |

### 3.3 LoRA vs full fine-tuning

```mermaid
flowchart TB
    subgraph Full["Full fine-tuning"]
        F1[Atualiza todos os pesos W] --> F2[Memória alta]
        F2 --> F3[Checkpoint gigante]
        F3 --> F4[Risco de esquecimento]
    end
    subgraph LoRA["LoRA / PEFT"]
        L1[Congela W0] --> L2[Treina apenas B e A]
        L2 --> L3[Memória baixa]
        L3 --> L4[Checkpoint minúsculo]
        L4 --> L5[Preserva conhecimento]
    end
```

## 4. Código: implementando LoRA

### 4.1 A camada LoRA

```python
import torch
import torch.nn as nn
import math

class CamadaLoRA(nn.Module):
    def __init__(self, d, k, r=16, alpha=16):
        super().__init__()
        self.d = d
        self.k = k
        self.r = r
        self.escala = alpha / r

        # Matriz original: congelada (sem gradiente)
        self.W0 = nn.Parameter(torch.randn(d, k), requires_grad=False)

        # Matrizes do adaptador: treináveis
        self.A = nn.Parameter(torch.randn(r, k) * 0.02)   # init gaussiana pequena
        self.B = nn.Parameter(torch.zeros(d, r))          # init zeros

    def forward(self, h):
        # h: (..., k)
        saida_base = h @ self.W0.T                      # W0 h
        saida_lora = (h @ self.A.T) @ self.B.T          # B (A h)
        return saida_base + self.escala * saida_lora
```

Pontos importantes do código:

1. `requires_grad=False` em `W0` **congela** a matriz original.
2. `A` inicia com valores pequenos, `B` com zeros, garantindo $\Delta W = 0$ no início.
3. A escala `alpha / r` controla a magnitude do adaptador.

### 4.2 Aplicando a várias camadas

```python
def aplicar_lora(modelo, r=16, alpha=16, alvo=["q_proj", "v_proj"]):
    """Substitui camadas lineares selecionadas por CamadaLoRA."""
    for nome, modulo in modelo.named_modules():
        if any(t in nome for t in alvo):
            pai = modelo
            partes = nome.split(".")
            for p in partes[:-1]:
                pai = getattr(pai, p)
            linear = getattr(pai, partes[-1])
            d, k = linear.weight.shape
            setattr(
                pai,
                partes[-1],
                CamadaLoRA(d, k, r=r, alpha=alpha),
            )
    return modelo
```

### 4.3 Treino: apenas os parâmetros LoRA


### 4.4 Pseudocódigo

```text
ALGORITMO LoRA(modelo, r, alpha):
    PARA cada camada linear alvo:
        congelar W0 (requires_grad = False)
        criar A ~ N(0, sigma^2)  de forma (r, k)
        criar B = 0               de forma (d, r)
        substituir forward: h' = W0 h + (alpha/r) B A h
    otimizador = AdamW(apenas parâmetros de A e B)
    treinar normalmente
    salvar apenas A e B (checkpoint pequeno)
```

## 6. Erros comuns com LoRA/PEFT

### 6.1 Esquecer de congelar os pesos base

Se `W0` continua com `requires_grad=True`, você está fazendo full fine-tuning disfarçado, e perde toda a economia. Sintoma: memória explode. Solução: garantir `requires_grad=False` no modelo base.

### 6.2 Inicializar B com valores não nulos

Se $B$ não começa em zero, o adaptador perturba o modelo antes do treino, causando instabilidade. Solução: sempre `torch.zeros` para $B$.

### 6.3 Posto $r$ mal escolhido

$r$ muito baixo ($r = 1$) pode não capturar a adaptação necessária; $r$ muito alto perde a eficiência. Solução: começar com $r = 8$ ou $16$ e ajustar.

### 6.4 Aplicar LoRA nas camadas erradas

Aplicar em todas as camadas pode ser caro; aplicar só na última pode ser insuficiente. Solução: começar em `q_proj` e `v_proj` (atenção), e expandir para MLP se necessário.

### 6.5 Esquecer o merge antes de compilar

Se o runtime não oferece suporte a adaptadores separados, pode ser necessário fundi-los aos pesos-base antes da conversão. Verifique os formatos de entrada e as instruções do runtime de destino antes de exportar.

## 7. Resumo

1. **PEFT** adapta o modelo treinando poucos parâmetros novos, com pesos base congelados.
2. **LoRA** fatora a atualização como $\Delta W = B A$, com posto $r \ll d, k$.
3. A passagem direta é $h' = W_0 h + \frac{\alpha}{r} B A h$.
4. A inicialização ($B = 0$) garante que o adaptador comece neutro.
5. A economia pode chegar a **centenas de vezes** em parâmetros treináveis.
7. Erros comuns: não congelar, inicializar $B$ errado, posto inadequado, camadas erradas e esquecer o merge.

A essência da LoRA: **não reforme a casa inteira — troque só a janela, e com pouquíssimas peças.**
