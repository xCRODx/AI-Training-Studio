# Gradientes e derivadas parciais

## Intuição: para onde a função "aponta"

Imagine que você está em uma montanha, no meio de uma névoa densa. Você não consegue ver o topo nem o vale, mas consegue sentir, com o pé, a inclinação do terreno em volta de você. Se o chão sobe para o norte e desce para o sul, você sabe que, para chegar ao topo, deve caminhar para o norte. Se quer descer até o vale, caminha para o sul.

Um **gradiente** faz exatamente isso com uma função de várias variáveis: ele diz, em cada ponto, qual é a direção em que a função **cresce mais rápido** — e, por consequência, qual direção tomar para **descer mais rápido**.


> "Se você quer descer uma montanha sem enxergar nada, dê sempre um passo na direção oposta à subida mais íngreme." — essa é a essência do gradiente descendente.

## Conceito: o que é uma derivada parcial

### Revisão em uma variável

Antes de derivadas parciais, lembre-se da derivada comum. Para $f(x) = x^2$, a derivada mede a taxa instantânea de variação:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} = 2x
$$

Ela responde: "se eu der um passo minúsculo $h$ em $x$, quanto $f$ muda, proporcionalmente a $h$?"

### Várias variáveis: derivada parcial

Quando a função tem várias entradas, por exemplo $f(x, y) = x^2 y + 3y$, precisamos decidir **qual** variável vamos mexer. A derivada parcial de $f$ em relação a $x$ trata todas as outras variáveis como **constantes**:

$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}
$$

Ou seja: é a derivada comum, mas apenas na direção de $x$, congelando $y$.

**Exemplo numérico passo a passo.** Seja $f(x, y) = x^2 y + 3y$.

1. Para $\frac{\partial f}{\partial x}$, tratamos $y$ como constante:
   - Derivada de $x^2 y$ em $x$ é $2x y$ (o $y$ acompanha como coeficiente).
   - Derivada de $3y$ em $x$ é $0$ (não há $x$).
   - Resultado: $\frac{\partial f}{\partial x} = 2xy$.

2. Para $\frac{\partial f}{\partial y}$, tratamos $x$ como constante:
   - Derivada de $x^2 y$ em $y$ é $x^2$.
   - Derivada de $3y$ em $y$ é $3$.
   - Resultado: $\frac{\partial f}{\partial y} = x^2 + 3$.

3. Avaliando no ponto $(x, y) = (2, 1)$:
   - $\frac{\partial f}{\partial x}(2,1) = 2 \cdot 2 \cdot 1 = 4$.
   - $\frac{\partial f}{\partial y}(2,1) = 2^2 + 3 = 7$.

Interpretação: no ponto $(2,1)$, se aumentarmos $x$ em uma unidade pequena, $f$ cresce cerca de 4 vezes essa variação; se aumentarmos $y$, $f$ cresce cerca de 7 vezes.

## O vetor gradiente

O gradiente reúne **todas** as derivadas parciais em um único vetor:

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)
$$

Para o exemplo acima:

$$
\nabla f = \left( 2xy,\; x^2 + 3 \right)
$$

No ponto $(2,1)$:

$$
\nabla f(2,1) = (4,\; 7)
$$

Em uma rede neural com milhões de parâmetros, o gradiente é simplesmente um vetor gigante com uma entrada para cada peso e cada viés.

### O gradiente aponta para o crescimento

O fato central do gradiente: **a direção de $\nabla f$ é a direção de maior crescimento local de $f$**. O sentido oposto, $-\nabla f$, é a direção de maior **decréscimo**.

Isso vem da fórmula de variação de primeira ordem. Para um deslocamento pequeno $\Delta w$:

$$
f(w + \Delta w) \approx f(w) + \nabla f(w) \cdot \Delta w
$$

O produto escalar $\nabla f(w) \cdot \Delta w$ é maximizado (para um tamanho fixo de $\Delta w$) quando $\Delta w$ aponta na mesma direção de $\nabla f(w)$. Portanto:

- $\Delta w = \nabla f(w)$ ⇒ máximo aumento.
- $\Delta w = -\nabla f(w)$ ⇒ máxima redução.

## O passo do gradiente descendente

Queremos minimizar a perda $L(w)$ em função dos pesos $w$. A regra de atualização é:

$$
w \leftarrow w - \eta \, \nabla L(w)
$$

Onde:

- $w$ é o vetor (ou tensor) de parâmetros;
- $\eta$ é a **taxa de aprendizado** (learning rate), um número positivo pequeno;
- $\nabla L(w)$ é o gradiente da perda em relação a $w$.

O sinal de menos garante que andamos **contra** o gradiente (morro abaixo). O $\eta$ controla o tamanho do passo: grande demais e saltamos o mínimo; pequeno demais e demoramos muito.

### Exemplo numérico de um passo

Considere a perda $L(w) = w^2$, com $w$ escalar. Sabemos que $\frac{dL}{dw} = 2w$.

Suponha $w = 3$ e $\eta = 0{,}1$:

1. Gradiente: $\nabla L(3) = 2 \cdot 3 = 6$.
2. Atualização: $w \leftarrow 3 - 0{,}1 \cdot 6 = 3 - 0{,}6 = 2{,}4$.

Repetindo:

| Passo | $w$ antes | $\nabla L = 2w$ | $w - \eta\nabla L$ |
|-------|-----------|-----------------|--------------------|
| 1     | 3,0       | 6,0             | 2,4                |
| 2     | 2,4       | 4,8             | 1,92               |
| 3     | 1,92      | 3,84            | 1,536              |

A cada passo, $w$ se aproxima de $0$, que é o mínimo de $L(w) = w^2$. Repare que o passo diminui conforme nos aproximamos do mínimo, porque o gradiente fica menor.

### Exemplo com duas variáveis

Seja $L(w_1, w_2) = w_1^2 + w_2^2$. O gradiente é:

$$
\nabla L = \left( \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2} \right) = (2w_1,\; 2w_2)
$$

Começando em $(w_1, w_2) = (2, -1)$ com $\eta = 0{,}2$:

1. $\nabla L = (2 \cdot 2,\; 2 \cdot (-1)) = (4, -2)$.
2. $w_1 \leftarrow 2 - 0{,}2 \cdot 4 = 2 - 0{,}8 = 1{,}2$.
3. $w_2 \leftarrow -1 - 0{,}2 \cdot (-2) = -1 + 0{,}4 = -0{,}6$.

O mínimo é a origem $(0,0)$, e estamos nos aproximando dela.

## O gradiente de um tensor tem o mesmo shape

Em redes neurais, os parâmetros não são vetores soltos: são **tensores** — matrizes de pesos $W \in \mathbb{R}^{m \times n}$, tensores de convolução, etc. Uma propriedade essencial que a maioria dos frameworks (PyTorch, JAX, etc.) respeita:

> O gradiente de um tensor sempre tem o **mesmo formato (shape)** do tensor original.

Se $W$ é uma matriz $3 \times 4$, então $\frac{\partial L}{\partial W}$ também é $3 \times 4$, e cada entrada $(\partial L / \partial W)_{ij}$ mede o quanto a perda muda ao variar o peso individual $W_{ij}$. Isso permite fazer a atualização elemento a elemento:

$$
W_{ij} \leftarrow W_{ij} - \eta \cdot \frac{\partial L}{\partial W_{ij}}
$$

Em notação tensorial compacta, escrevemos simplesmente:

$$
W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}
$$

onde a subtração e a multiplicação são feitas componente a componente (broadcasting).

## Código: autograd na prática

O cálculo automático de gradientes (autograd) elimina a necessidade de derivar manualmente. Veja um exemplo em PyTorch:

```python
import torch

# Parâmetros que queremos treinar
w = torch.tensor([3.0, -1.0], requires_grad=True)

# Taxa de aprendizado
eta = 0.1

# Função de perda: L(w) = soma dos quadrados
def perda(w):
    return (w ** 2).sum()

for passo in range(5):
    L = perda(w)          # calcula a perda
    L.backward()          # computa grad(w) automaticamente
    with torch.no_grad():
        w -= eta * w.grad # passo do gradiente descendente
        w.grad.zero_()    # zera o gradiente acumulado
    print(f"passo {passo}: w = {w.tolist()}, L = {L.item():.4f}")
```

Saída esperada (valores aproximados):

```text
passo 0: w = [2.4, -0.8], L = 6.4000
passo 1: w = [1.92, -0.64], L = 4.0960
passo 2: w = [1.536, -0.512], L = 2.6214
passo 3: w = [1.2288, -0.4096], L = 1.6777
passo 4: w = [0.98304, -0.32768], L = 1.0737
```

Repare que, para $L(w) = \|w\|^2$, a atualização $w \leftarrow w - \eta \cdot 2w = (1 - 2\eta) w$ encolhe cada componente por um fator $1 - 2\eta = 0{,}8$ a cada passo. É exatamente o que a saída mostra.

### O que o `backward()` faz por baixo dos panos

O autograd constrói um **grafo computacional**: cada operação vira um nó, e os valores fluem "para frente" no cálculo da perda. Quando chamamos `backward()`, o framework percorre esse grafo **de trás para frente**, aplicando a regra da cadeia (ver o documento sobre a regra da cadeia) e acumulando o gradiente em cada parâmetro. É por isso que precisamos de `w.grad.zero_()`: o PyTorch **acumula** gradientes por padrão, então, sem zerar, somaríamos gradientes de passos anteriores.

## Pseudocódigo do treinamento

```text
entrada: parâmetros w, taxa eta, número de passos T
para t = 1 até T:
    L = perda(w)            # forward
    g = gradiente(L, w)     # backward (autograd)
    w = w - eta * g         # passo de descida
    devolver w
```

## Erros comuns

- **Esquecer o sinal de menos.** Usar $w \leftarrow w + \eta \nabla L$ faz a **subida** do gradiente, maximizando a perda em vez de minimizar. O modelo diverge.
- **Não zerar o gradiente.** Em PyTorch, `backward()` acumula. Sem `zero_()`, os gradientes somam-se entre passos e a descida fica errada.
- **Taxa de aprendizado grande.** Com $\eta$ grande, o passo pula o mínimo e a perda oscila ou explode. Com $\eta$ pequeno, o treino é lento demais.
- **Confundir derivada parcial com derivada total.** Ao derivar em $x$, não trate $y$ como função de $x$ quando elas são independentes — congele as demais variáveis.
- **Achar que o gradiente é um número.** Para funções de várias variáveis, o gradiente é um **vetor** (ou tensor). Cada componente é uma derivada parcial.
- **Esperar que o gradiente aponte para o mínimo.** Ele aponta para o **crescimento local** mais íngreme; o mínimo está na direção **oposta** (e somente localmente).

## Resumo

- A derivada parcial $\frac{\partial f}{\partial x}$ mede a variação de $f$ ao mexer apenas em $x$, congelando as demais variáveis.
- O gradiente $\nabla f$ é o vetor de todas as derivadas parciais e aponta para a direção de **maior crescimento**.
- O gradiente descendente atualiza os pesos por $w \leftarrow w - \eta \nabla L(w)$.
- O gradiente de um tensor tem o **mesmo shape** do tensor, permitindo atualização componente a componente.
- O autograd (PyTorch/JAX) calcula gradientes via grafo computacional e regra da cadeia.
