# Camadas lineares e embeddings

> "Tudo em um modelo de linguagem é uma matriz. O `nn.Linear` transforma vetores; o embedding transforma tokens em vetores. São as duas portas de entrada do mundo numérico."

---

## 1. Intuição: transformar números em números

Um modelo de linguagem só entende números. Para ele processar um texto, cada token (palavra ou subpalavra) precisa ser convertido em um **vetor de números reais**. Esse vetor é o que chamamos de *embedding*.


---

## 2. A camada linear (`nn.Linear`)

### 2.1 Definição matemática

Uma camada linear com $d_{\text{in}}$ entradas e $d_{\text{out}}$ saídas é definida por uma matriz de pesos $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ e um vetor de viés $\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$:

$$
\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}
$$

Expandindo para uma única saída $j$:

$$
y_j = \sum_{i=1}^{d_{\text{in}}} w_{j,i} \cdot x_i + b_j
$$

Cada linha da matriz $\mathbf{W}$ é o vetor de pesos de um neurônio. Logo, uma camada linear com $d_{\text{out}}$ saídas contém exatamente $d_{\text{out}}$ neurônios artificiais, como vimos no documento do neurônio.

### 2.2 Por que "linear" se o modelo é não-linear?

A camada `nn.Linear` é chamada de linear porque sua operação (multiplicação de matriz mais soma de vetor) é linear. A não-linearidade é introduzida **separadamente** pelas funções de ativação. Por isso, em código, vemos sempre o padrão:

```text
linear -> ativação -> linear -> ativação -> ...
```

O `nn.Linear` puro não faz nada não-linear sozinho.

### 2.3 Exemplo numérico completo

Considere uma camada com $d_{\text{in}} = 3$ e $d_{\text{out}} = 2$:

$$
\mathbf{W} = \begin{bmatrix} 0.5 & -0.2 & 1.0 \\ 0.1 & 0.4 & -0.3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 0.0 \\ 0.2 \end{bmatrix}
$$

Entrada:

$$
\mathbf{x} = \begin{bmatrix} 1.0 \\ 2.0 \\ 0.5 \end{bmatrix}
$$

Cálculo de $y_1$:

$$
y_1 = (0.5 \cdot 1.0) + (-0.2 \cdot 2.0) + (1.0 \cdot 0.5) + 0.0 = 0.6
$$

Cálculo de $y_2$:

$$
y_2 = (0.1 \cdot 1.0) + (0.4 \cdot 2.0) + (-0.3 \cdot 0.5) + 0.2 = 0.95
$$

Resultado:

$$
\mathbf{y} = \begin{bmatrix} 0.6 \\ 0.95 \end{bmatrix}
$$

### 2.4 Número de parâmetros

O número de parâmetros de uma camada linear é:

$$
\text{parâmetros} = d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}}
$$

O termo $d_{\text{in}} \cdot d_{\text{out}}$ vem dos pesos e $d_{\text{out}}$ vem do viés. No exemplo acima: $3 \cdot 2 + 2 = 8$ parâmetros.

---

## 3. Embeddings: transformar tokens em vetores

### 3.1 O problema dos tokens

Um texto é uma sequência de tokens $t_1, t_2, \dots, t_L$, onde cada token é um inteiro (índice no vocabulário). Esses inteiros não carregam significado: o token $17$ não é "mais parecido" com o token $18$ do que com o token $9000$. Precisamos de uma representação densa e **aprendível**.

### 3.2 A tabela de embeddings

Um embedding é, na prática, uma **tabela de consulta** (lookup table): uma matriz $\mathbf{E} \in \mathbb{R}^{V \times d}$, onde:

- $V$ é o tamanho do vocabulário (ex.: 50.000 tokens).
- $d$ é a dimensão do modelo (ex.: 512 ou 1024).

Cada linha $\mathbf{E}_t$ é o vetor de embedding do token $t$. Converter um token em vetor é simplesmente indexar a linha:

$$
\mathbf{x}_t = \mathbf{E}_{t, :}
$$

### 3.3 Exemplo numérico

Suponha vocabulário $V = 5$ e dimensão $d = 3$. A tabela de embeddings é:

$$
\mathbf{E} = \begin{bmatrix}
0.10 & 0.20 & 0.30 \\
0.40 & 0.50 & 0.60 \\
0.70 & 0.80 & 0.90 \\
1.00 & 1.10 & 1.20 \\
1.30 & 1.40 & 1.50
\end{bmatrix}
$$

Para o token de índice $2$, o embedding é a terceira linha:

$$
\mathbf{x}_2 = \begin{bmatrix} 0.70 & 0.80 & 0.90 \end{bmatrix}
$$

### 3.4 Embeddings são aprendidos

No início, a tabela $\mathbf{E}$ é inicializada aleatoriamente. Durante o treinamento, o gradiente descendente ajusta cada linha para que tokens semanticamente parecidos fiquem **próximos** no espaço vetorial. Esse é o famoso efeito de "rei - homem + mulher = rainha", capturado por relações geométricas entre vetores.

### 3.5 Embedding de posição

Além do conteúdo do token, o modelo precisa saber a **posição** dele na sequência. Por isso, adiciona-se um embedding posicional:

$$
\mathbf{h}_t = \mathbf{E}_{t,:} + \mathbf{P}_{t,:}
$$

onde $\mathbf{P}_{t,:}$ é o vetor de posição do índice $t$. No caso do RetNet, a informação posicional também entra de forma implícita pelo mecanismo de retenção, mas o embedding posicional (ou rotação posicional) ainda é uma peça importante.

---

## 4. Shapes típicos em um modelo de linguagem

Aqui estão as dimensões mais comuns em arquiteturas tipo RetNet/Transformer:

| Componente | Shape | Descrição |
|---|---|---|
| Vocabulário | $V$ | Número de tokens (ex.: 32k, 50k, 128k) |
| Dimensão do modelo | $d$ | Largura dos embeddings (ex.: 512, 1024, 4096) |
| Embedding de tokens | $\mathbb{R}^{V \times d}$ | Tabela de lookup |
| Entrada de um lote | $\mathbb{R}^{B \times L \times d}$ | $B$ sequências de comprimento $L$ |
| Matriz de pesos linear | $\mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ | Pesos de uma camada |
| Logits de saída | $\mathbb{R}^{B \times L \times V}$ | Pontuação por token do vocabulário |

### 4.1 Exemplo concreto

Para um modelo com $V = 50.000$ e $d = 1024$, a tabela de embeddings tem:

$$
50.000 \cdot 1024 = 51.200.000
$$

Ou seja, **51,2 milhões de parâmetros apenas na tabela de embeddings**. Isso mostra por que a quantização (como a do BitNet, que reduz cada peso de 32 bits para $\{-1, 0, +1\}$ ou a 2 bits) tem impacto enorme no tamanho do modelo.

---

## 5. Código PyTorch

### 5.1 Criando uma camada linear

```python
import torch
import torch.nn as nn

linear = nn.Linear(in_features=3, out_features=2)
print("Matriz de pesos:", linear.weight.shape)  # torch.Size([2, 3])
print("Vies:", linear.bias.shape)               # torch.Size([2])

x = torch.tensor([[1.0, 2.0, 0.5]])
y = linear(x)
print("Saida:", y)
```

### 5.2 Criando um embedding

```python
import torch
import torch.nn as nn

vocab = 50000
dim = 1024

embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=dim)
print("Tabela:", embedding.weight.shape)  # torch.Size([50000, 1024])

tokens = torch.tensor([3, 42, 777])       # 3 tokens
vetores = embedding(tokens)
print("Vetores:", vetores.shape)          # torch.Size([3, 1024])
```

### 5.3 Montando o início de um modelo


---

## 6. Posição no modelo (diagrama Mermaid)

```mermaid
graph TD
    Tokens[Tokens: inteiros<br/>[3, 42, 777]] --> Emb[Embedding<br/>E em R^Vxd]
    Emb --> AddPos[+ Embedding posicional]
    AddPos --> Norm1[LayerNorm]
    Norm1 --> Ret[Retenção RetNet<br/>Q,K,V via Linear]
    Ret --> Norm2[LayerNorm]
    Norm2 --> FF[FeedForward<br/>Linear -> GELU -> Linear]
    FF --> Head[Linear de saída<br/>d -> V]
    Head --> Softmax[Softmax sobre vocabulário]
    Softmax --> Pred[Próximo token]
```

O diagrama mostra exatamente onde o `nn.Linear` e o `nn.Embedding` entram:

- O **embedding** fica na entrada, convertendo inteiros em vetores.
- Os **`nn.Linear`** aparecem em três lugares: nas projeções Q/K/V da retenção, no *feed-forward* (com GELU) e na cabeça de saída.
- A **softmax** aparece apenas no final, sobre o vocabulário.

---

## 8. Erros comuns

### 8.1 Esquecer o viés (ou incluí-lo duas vezes)

O `nn.Linear(in, out)` já inclui o viés por padrão (`bias=True`). Se você também somar manualmente um viés, estará duplicando. Use `bias=False` apenas quando aplicar sua própria normalização.

### 8.2 Confundir `embedding_dim` com tamanho do vocabulário

`nn.Embedding(num_embeddings=V, embedding_dim=d)` cria uma tabela de shape $(V, d)$. Inverter esses dois valores muda completamente o modelo e costuma gerar erros de shape em cascata.

### 8.3 Não transpor pesos

Ao implementar uma camada linear manualmente, lembre que a convenção é $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, e não o contrário. Trocar as dimensões gera um produto matriz-vetor incorreto.

### 8.4 Esquecer o reshape do lote

Muitas vezes a entrada é $[B, L, d]$ e a camada linear espera $[*, d]$. Em PyTorch, `nn.Linear` opera na última dimensão automaticamente, mas ao escrever código manual é fácil errar o *reshape*.

### 8.5 Ignorar a escala de inicialização

A inicialização dos pesos do `nn.Linear` importa. Inicializar pesos com valores grandes pode saturar ativações; valores pequenos podem retardar o aprendizado. Inicializações como Xavier/He são o padrão.

---

## 9. Resumo

- `nn.Linear` aplica $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$: é a camada fundamental, contendo $d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}}$ parâmetros.
- Embedding é uma tabela de lookup $\mathbf{E} \in \mathbb{R}^{V \times d}$ que converte inteiros de token em vetores aprendíveis.
- Shapes típicos: entrada $[B, L, d]$, pesos $[d_{\text{out}}, d_{\text{in}}]$, logits $[B, L, V]$.
- A tabela de embeddings costuma ser a maior parte dos parâmetros de um modelo de linguagem, o que torna a quantização especialmente valiosa.

---
