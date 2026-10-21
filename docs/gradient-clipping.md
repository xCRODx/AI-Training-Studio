# Gradient clipping

## Intuição: o freio de emergência da descida

Imagine de novo a descida da montanha na névoa. Você sente com o pé a inclinação e dá passos proporcionais a ela. Mas em uma descida muito íngreme — um penhasco — a inclinação é tão grande que o passo "proporcional" vira um salto. Você se desequilibra, rola, e de repente está mais longe do vale do que estava antes.

Em redes neurais, o "penhasco" é um ponto em que o gradiente explode: a inclinação da perda em relação a algum peso é gigantesca. O passo proporcional a esse gradiente joga o peso para uma região absurda, de onde o treino dificilmente volta. O **gradient clipping** é o freio: ele limita o tamanho do gradiente a um teto, mantendo os passos dentro do seguro.


> "Se o gradiente é o acelerador, o clipping é o limitador de velocidade: deixa o treino correr, mas nunca perder o controle."

## Conceito: o que é a norma do gradiente

O gradiente de uma rede com $N$ parâmetros é um vetor gigante:

$$
\nabla L = \left( \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \dots, \frac{\partial L}{\partial w_N} \right)
$$

Para saber se esse vetor é "grande", medimos seu comprimento — a **norma** (tamanho). A norma mais comum é a norma L2 (euclidiana):

$$
\| \nabla L \|_2 = \sqrt{ \sum_{i=1}^{N} g_i^2 }
$$

Onde $g_i = \frac{\partial L}{\partial w_i}$ é o gradiente do $i$-ésimo parâmetro. Em Python/PyTorch, somamos os quadrados de todos os elementos do gradiente e tiramos a raiz:

$$
\| g \| = \sqrt{ g_1^2 + g_2^2 + \dots + g_N^2 }
$$

Uma norma pequena significa que a perda está quase plana (perto de um mínimo); uma norma enorme significa que há uma inclinação violenta em algum lugar.

## Por que os gradientes explodem

O gradiente de uma rede profunda é o produto de várias "jacobianas" (derivadas de cada camada) pela regra da cadeia. Se cada camada multiplica o sinal por um fator maior que $1$, o produto cresce exponencialmente com a profundidade:

$$
g \approx J_1 \cdot J_2 \cdot \dots \cdot J_L
$$


As causas mais comuns de explosão são:

1. **Pesos grandes.** Se os pesos crescem, as ativações crescem, e as derivadas crescem juntas.
2. **Learning rate alta.** Passos grandes levam os pesos para regiões de gradiente ainda maior, num ciclo vicioso.
3. **Perda muito íngreme.** Funções como a exponencial ou pontos de `softmax` extremos geram gradientes enormes localmente.
5. **Acúmulo de gradientes.** Quando vários microbatches são somados antes do passo, também se somam suas contribuições ao gradiente. O limite de clipping precisa ser aplicado no ponto adequado desse ciclo.

### Um exemplo numérico de explosão

Considere uma perda em que o gradiente em relação a um peso é $g = 4 \cdot w^3$. Começando em $w = 1$ com $\eta = 0{,}1$:

| Passo | $w$ antes | $g = 4w^3$ | $\Delta w = \eta g$ | $w$ depois |
|-------|-----------|------------|----------------------|------------|
| 1     | 1,00      | 4,00       | 0,40                 | 1,40       |
| 2     | 1,40      | 10,98      | 1,10                 | 2,50       |
| 3     | 2,50      | 62,50      | 6,25                 | 8,75       |
| 4     | 8,75      | 2679,69    | 267,97               | 276,72     |

Em apenas 4 passos, o peso saiu de $1$ para $276$, e o gradiente de $4$ para $2679$. A atualização $\Delta w$ cresce sem controle. É exatamente isso que o clipping impede.

## O clipping por norma

A técnica mais usada é o **clip por norma global**. A ideia é simples: calcular a norma $\|g\|$ de **todos** os gradientes juntos (a "norma global") e, se ela passar de um teto $C$, **encolher o gradiente inteiro** proporcionalmente, para que a norma fique exatamente em $C$:

$$
\tilde{g} =
\begin{cases}
g, & \text{se } \|g\| \leq C \\[6pt]
g \cdot \dfrac{C}{\|g\|}, & \text{se } \|g\| > C
\end{cases}
$$

O fator de escala é $\frac{C}{\|g\|} < 1$. Como todos os componentes são multiplicados pelo mesmo fator, a **direção** do gradiente é preservada; só o **tamanho** é limitado. É como reduzir a velocidade do carro sem mudar a rota.


### Exemplo numérico do clipping

Suponha que em um passo a rede tenha três parâmetros com gradientes $g = (3, 4, 12)$. A norma é:

$$
\|g\| = \sqrt{3^2 + 4^2 + 12^2} = \sqrt{9 + 16 + 144} = \sqrt{169} = 13
$$


$$
\frac{C}{\|g\|} = \frac{0{,}1}{13} \approx 0{,}00769
$$

O gradiente clipado vira:

$$
\tilde{g} = (3 \cdot 0{,}00769,\; 4 \cdot 0{,}00769,\; 12 \cdot 0{,}00769) \approx (0{,}0231,\; 0{,}0308,\; 0{,}0923)
$$

A nova norma é:

$$
\sqrt{0{,}0231^2 + 0{,}0308^2 + 0{,}0923^2} \approx \sqrt{0{,}000533 + 0{,}000949 + 0{,}008519} \approx \sqrt{0{,}010001} \approx 0{,}1
$$

Exatamente o teto $C$. Repare que a proporção $3:4:12$ entre os gradientes foi mantida — só o tamanho total encolheu.

### Um caso que não clipeia

Se os gradientes forem $g = (0{,}02, 0{,}03, 0{,}04)$, a norma é:

$$
\|g\| = \sqrt{0{,}0004 + 0{,}0009 + 0{,}0016} = \sqrt{0{,}0029} \approx 0{,}0538
$$

Como $0{,}0538 < 0{,}1$, nada muda: $\tilde{g} = g$. O clipping só age quando a norma **excede** o teto, deixando gradientes pequenos intocados.

## Acompanhar a norma do gradiente


```python
max_norm = 1.0  # valor ilustrativo; deve ser validado para o treino
grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
last_grad_norm = grad_norm_tensor.item()
```

O detalhe crucial: `clip_grad_norm_` **retorna a norma original** (antes do clipping). Assim, o valor retornado pode ser $13{,}0$ mesmo com `max_norm = 0.1$: o gradiente foi redimensionado para o teto, mas a função reporta a norma que tinha antes.

### Como ler a métrica

1. **Norma abaixo do teto.** Se a norma fica abaixo de `max_norm`, o clipping não está agindo. Isso, por si só, não prova que todo o treino está estável.

2. **Norma acima do teto, mas estável.** Picos ocasionais acima do teto são normais e esperados — é o clipping fazendo o trabalho dele, cortando o excesso.

3. **Norma crescendo sem parar.** Se a norma sobe passo após passo (de $0{,}1$ para $1$, depois $10$...), o gradiente está divergindo. O clipping segura o passo, mas o sinal subjacente está piorando — pode ser hora de reduzir a taxa de aprendizado ou checar os dados.

4. **Norma caindo a zero.** Uma norma que desaba para valores minúsculos pode indicar que o treino estagnou (gradientes quase nulos) ou que algo no cálculo quebrou.

## Código: clipping na prática

### Uma implementação didática

Sem a biblioteca, dá para reescrever o clip por norma global:

```python
import math

def clip_grad_norm(params, max_norm):
    total_norm = 0.0
    for p in params:
        if p.grad is None:
            continue
        total_norm += (p.grad.detach().norm(2).item()) ** 2
    total_norm = math.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(scale)

    return total_norm
```

Pontos-chave da implementação:

- O `** 2` acumula a soma dos quadrados; a raiz no fim dá a norma L2.
- O `1e-6` no denominador evita divisão por zero quando `total_norm` é exatamente zero.
- O `.detach()` garante que a própria operação de clipping não entre no grafo de autograd.
- O retorno é a norma **original**, antes do clipping.

### Verificando o exemplo numérico

```python
import torch

a = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([4.0], requires_grad=True)
c = torch.tensor([12.0], requires_grad=True)
params = [a, b, c]

for p in params:
    p.grad = torch.ones_like(p)

# Gradientes: (3, 4, 12) -> norma 13
a.grad = torch.tensor([3.0])
b.grad = torch.tensor([4.0])
c.grad = torch.tensor([12.0])

norm = clip_grad_norm(params, max_norm=0.1)
print(f"norma original: {norm:.4f}")
print(f"gradientes clipados: {a.grad.item():.4f}, {b.grad.item():.4f}, {c.grad.item():.4f}")
```

Saída esperada:

```text
norma original: 13.0000
gradientes clipados: 0.0231, 0.0308, 0.0923
```

Bate com o exemplo numérico anterior.

## Pseudocódigo do clipping

```text
entrada: gradientes g (de todos os parâmetros), teto C
norma = raiz(soma dos quadrados de todos os g)
se norma > C:
    fator = C / norma
    para cada parâmetro:
        g = g * fator
devolver norma_original
```

## Outras formas de clipping (contexto)

Além do clip por norma global, existem variações:

1. **Clip por valor.** Limita cada elemento do gradiente a um intervalo fixo, por exemplo $\min(\max(g_i, -v), v)$. Corta picos individuais, mas pode distorcer a direção.
2. **Clip por norma por parâmetro.** Calcula a norma de cada tensor separadamente e clipa cada um. Útil quando as camadas têm escalas muito diferentes.

O clip por norma global é o mais comum em modelos de linguagem porque preserva a direção do gradiente e é robusto a redes com muitas camadas de escalas variadas.

## Erros comuns

- **Achar que clipping altera a direção do gradiente.** O clip por norma global multiplica **todos** os componentes pelo mesmo fator, preservando a direção. Só o clip por valor distorce a direção.
- **Confundir a norma retornada com a norma clipada.** `clip_grad_norm_` retorna a norma **original**, não o teto aplicado.
- **Clipar a cada microbatch em vez de a cada passo.** Com acúmulo de gradientes, o clip deve ser aplicado na soma acumulada, senão o passo final pode ultrapassar o teto.
- **Clipar depois do `optimizer.step()`.** O clipping precisa acontecer **antes** do passo; depois é tarde, o estrago já foi feito.
- **Usar `max_norm` baixo demais.** Um teto minúsculo corta gradientes com frequência, podendo retardar o aprendizado ou estagná-lo.
- **Ignorar a métrica.** A "Norma do gradiente" é um termômetro. Uma norma subindo sem parar pede ação (reduzir $\eta$, checar dados), mesmo que o clipping ainda esteja segurando os passos.
- **Esquecer de zerar gradientes.** Como o clip usa os gradientes acumulados, um `zero_grad()` faltando faz a norma crescer artificialmente e o clipping cortar demais.

## Resumo

- A norma do gradiente $\|g\| = \sqrt{\sum g_i^2}$ mede o tamanho do vetor de gradiente; normas enormes indicam inclinações violentas na perda.
- Gradientes podem crescer por vários motivos, como produtos de jacobianas, escala dos parâmetros, taxa de aprendizado ou características da perda. Estimadores aproximados também podem interagir com o processo.
- O clip por norma global multiplica o gradiente inteiro por $\frac{C}{\|g\|}$ quando $\|g\| > C$, limitando o tamanho sem mudar a direção.
- `clip_grad_norm_` retorna a norma **pré-clip** — por isso ela pode exceder o limite definido.
