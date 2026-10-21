# Álgebra linear essencial

## Intuição: matrizes como transformações de espaço

Imagine que você tem uma folha de borracha esticada. Você pode **esticar**, **girar**, **achatar** ou **espelhar** essa folha. Uma **matriz** é uma forma de descrever exatamente isso: uma transformação que leva vetores de um lugar para outro.

Na álgebra linear, um vetor $x$ que entra em uma matriz $A$ sai transformado em outro vetor:

$$
y = A x
$$


> Matrizes são as "máquinas" da álgebra linear: elas transformam vetores. Entender os seus ingredientes (inversa, determinante, autovalores, SVD) é entender o que essas máquinas fazem com os dados.

## Matriz identidade: o "não fazer nada"

A matriz identidade $I_n$ é a matriz que, quando multiplica qualquer vetor, o devolve **inalterado**:

$$
I_n x = x
$$

Para $n = 2$:

$$
I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

Ela é o "elemento neutro" da multiplicação de matrizes, assim como $1$ é o neutro da multiplicação de números: $A I = I A = A$.

**Exemplo numérico:**

$$
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 7 \end{bmatrix} = \begin{bmatrix} 1 \cdot 3 + 0 \cdot 7 \\ 0 \cdot 3 + 1 \cdot 7 \end{bmatrix} = \begin{bmatrix} 3 \\ 7 \end{bmatrix}
$$

Em redes, a identidade aparece como inicialização de certas camadas (como a "conexão residual" implícita), e como referência para medir o quanto uma matriz "se afasta" de não fazer nada.

## Matriz transposta: espelhar a matriz

A transposta $A^T$ troca linhas por colunas. Para uma matriz $2 \times 2$:

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \Rightarrow \quad A^T = \begin{bmatrix} a & c \\ b & d \end{bmatrix}
$$

De forma geral, $(A^T)_{ij} = A_{ji}$.

**Exemplo numérico:**

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad A^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}
$$

**Propriedades úteis:**

- $(A^T)^T = A$
- $(AB)^T = B^T A^T$ (a ordem **inverte**)

**Onde aparece no modelo.** No backpropagation, o gradiente de uma camada linear $z = W x$ em relação a $x$ é dado por $W^T \nabla_z L$. A transposta é o "caminho de volta" do gradiente. Também aparece no cálculo de atenção $Q K^T$: transpor a chave $K$ alinha as dimensões para o produto escalar entre consultas e chaves.

## Matriz inversa: desfazer a transformação

Se uma matriz $A$ transforma $x$ em $y$, a **inversa** $A^{-1}$ desfaz essa transformação:

$$
A^{-1} A = A A^{-1} = I
$$

Intuitivamente: se $A$ "esticou e girou", $A^{-1}$ "encolhe e gira de volta", deixando tudo como estava.

**Exemplo resolvido 2×2.** Para

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

a inversa (quando $ad - bc \neq 0$) é:

$$
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

**Com números.** Seja

$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}
$$

O determinante é $ad - bc = 2 \cdot 1 - 1 \cdot 1 = 1$. Então:

$$
A^{-1} = \frac{1}{1} \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}
$$

Verificação:

$$
A A^{-1} = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 2 \cdot 1 + 1 \cdot (-1) & 2 \cdot (-1) + 1 \cdot 2 \\ 1 \cdot 1 + 1 \cdot (-1) & 1 \cdot (-1) + 1 \cdot 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

Confere.

**Intuição importante.** Nem toda matriz tem inversa. Se a transformação "achata" o espaço (como projetar tudo em uma linha), não dá para desfazer — informação foi perdida. Essas matrizes são **singulares** e têm determinante $0$. É por isso que o determinante (próxima seção) é o termômetro da invertibilidade.

## Determinante: quanto a transformação "estica" a área

O determinante $\det(A)$ mede o **fator de escala de área (ou volume)** da transformação. Se $A$ transforma um quadrado de área 1, o resultado tem área $|\det(A)|$.

Para uma matriz $2 \times 2$:

$$
\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
$$

**Interpretações do determinante:**

- $\det(A) > 0$: a transformação preserva orientação (não espelha).
- $\det(A) < 0$: a transformação espelha (inverte orientação).
- $\det(A) = 0$: a transformação "achata" o espaço (perde uma dimensão), e $A$ não tem inversa.
- $|\det(A)|$: fator de escala de área/volume.

**Exemplo numérico.** Seja

$$
A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}
$$

$$
\det(A) = 3 \cdot 2 - 1 \cdot 0 = 6
$$

A transformação multiplica áreas por 6. Um quadrado unitário vira um paralelogramo de área 6. Como $6 > 0$, não há espelhamento.


## Autovalores e autovetores: as "direções privilegiadas"

Quando uma matriz multiplica um vetor, em geral o resultado aponta para outra direção. Mas existem vetores especiais, os **autovetores**, que apenas são **esticados/encolhidos** sem mudar de direção:

$$
A v = \lambda v
$$

Aqui $v$ é o autovetor e $\lambda$ é o **autovalor** (o fator de escala naquela direção).

**Intuição.** Pense numa matriz como um esticamento de borracha. Existem direções em que a borracha é puxada "em linha reta", sem torção — são os autovetores. O quanto cada uma estica é o autovalor correspondente.

### Como encontrar (passo a passo)

Queremos resolver $A v = \lambda v$, ou seja, $(A - \lambda I) v = 0$. Para existir solução não nula, $A - \lambda I$ deve ser singular, logo:

$$
\det(A - \lambda I) = 0
$$

**Exemplo resolvido 2×2.** Seja

$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

1. Monte $A - \lambda I$:

   $$
   A - \lambda I = \begin{bmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{bmatrix}
   $$

2. Calcule o determinante e iguale a zero:

   $$
   \det(A - \lambda I) = (2-\lambda)^2 - 1 = 0
   $$

   $$
   (2-\lambda)^2 = 1 \;\Rightarrow\; 2 - \lambda = \pm 1 \;\Rightarrow\; \lambda = 1 \;\text{ou}\; \lambda = 3
   $$

3. Encontre os autovetores:
   - Para $\lambda = 3$: resolva $(A - 3I)v = 0$:
     $$
     \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = 0 \;\Rightarrow\; v_1 = v_2
     $$
     Autovetor: $v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.
   - Para $\lambda = 1$: resolva $(A - I)v = 0$:
     $$
     \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = 0 \;\Rightarrow\; v_1 = -v_2
     $$
     Autovetor: $v = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$.

**Interpretação.** A matriz estica a direção $\begin{bmatrix}1\\1\end{bmatrix}$ por 3, e a direção $\begin{bmatrix}1\\-1\end{bmatrix}$ por 1. Essas duas direções são as "privilegiadas" da transformação.

### Onde aparece no modelo

Autovalores aparecem em muitos lugares:

- **Estabilidade do treinamento.** Se os autovalores da matriz de pesos (ou da Hessiana da perda) são grandes, pequenas mudanças na entrada produzem grandes mudanças na saída — isso pode causar gradientes explosivos. Autovalores pequenos levam a gradientes que desaparecem.
- **Normalização.** Camadas como LayerNorm/RMSNorm buscam manter as ativações com escala controlada, combatendo autovalores extremos.
- **Atenção.** A matriz de atenção pode ser vista como uma combinação de projeções; seus autovalores descrevem como a informação flui entre tokens.

## Decomposição em valores singulares (SVD)

O SVD é uma forma de **decompor qualquer matriz** em três peças fáceis de entender:

$$
A = U \Sigma V^T
$$

Onde:

- $U$: matriz ortogonal (colunas são direções de saída).
- $\Sigma$: matriz diagonal com os **valores singulares** $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$ (não negativos).
- $V^T$: matriz ortogonal transposta (colunas de $V$ são direções de entrada).

**Intuição.** Toda transformação linear pode ser vista como três passos:

1. **Girar** o espaço de entrada (por $V^T$),
2. **Esticar** ao longo dos eixos (por $\Sigma$, que é diagonal),
3. **Girar** de novo (por $U$).

O SVD nos diz as direções em que a matriz mais amplifica (valores singulares grandes) e as que mais suprime (valores singulares pequenos ou zero).

### Exemplo numérico 2×2

Seja

$$
A = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}
$$

Essa matriz já é diagonal, então seu SVD é imediato: $U = I$, $\Sigma = A$ (com valores singulares $3$ e $1$), $V^T = I$.

A matriz estica a direção $x$ por 3 e deixa a direção $y$ como está. Os valores singulares $3$ e $1$ capturam exatamente isso.

Para uma matriz não diagonal como

$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

(usada na seção de autovalores), o SVD relaciona-se com os autovalores: para matrizes simétricas, os valores singulares são os **valores absolutos** dos autovalores, ou seja, $3$ e $1$ (aqui os autovalores já são positivos). As direções $U$ e $V$ coincidem com os autovetores.

### Onde aparece no modelo

O SVD tem aplicações importantes em aprendizado de máquina:

- **Low-Rank Adaptation (LoRA).** Técnicas de fine-tuning eficiente decompõem atualizações de pesos em $B A$ com posto baixo — ideia diretamente ligada ao SVD (que nos dá a melhor aproximação de posto baixo de uma matriz).
- **Redução de dimensionalidade.** O SVD é o coração da PCA: manter apenas os maiores valores singulares preserva a maior parte da "energia" dos dados.
- **Compressão e análise de pesos.** Inspecionar os valores singulares de uma matriz de pesos revela se a camada é "redundante" (muitos valores singulares próximos de zero), o que inspira poda (pruning) e quantização.
- **Inicialização.** Inicializações ortogonais (como manter $U, V$ ortogonais) ajudam a estabilizar o treinamento.

## Onde cada conceito aparece no modelo (tabela-resumo)

|----------|---------|----------|-------------------------------|
| Identidade | $I$ | "não faz nada" | conexões residuais, inicialização |
| Transposta | $A^T$ | espelhar linhas/colunas | backprop ($W^T \nabla L$), atenção ($Q K^T$) |
| Inversa | $A^{-1}$ | desfazer a transformação | resolver sistemas, whitening |
| Determinante | $\det(A)$ | fator de escala de área | detectar singularidade, mudança de variável |
| Autovalores/vetores | $\lambda, v$ | direções que só esticam | estabilidade do gradiente, análise de atenção |
| SVD | $U \Sigma V^T$ | girar → esticar → girar | LoRA, PCA, compressão/poda de pesos |

## Código: manipulando os conceitos

Veja como calcular cada conceito em Python (com NumPy) para uma matriz 2×2:

```python
import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, 2.0]])

# Identidade
I = np.eye(2)

# Transposta
At = A.T
print("A^T =\n", At)

# Determinante
det = np.linalg.det(A)
print("det(A) =", det)  # 3.0

# Inversa
Ainv = np.linalg.inv(A)
print("A^{-1} =\n", Ainv)

# Verificacao: A . A^{-1} = I
print("A . A^{-1} =\n", A @ Ainv)

# Autovalores e autovetores
valores, vetores = np.linalg.eig(A)
print("autovalores =", valores)      # [3. 1.]
print("autovetores =\n", vetores)

# SVD
U, S, Vt = np.linalg.svd(A)
print("U =\n", U)
print("valores singulares =", S)     # [3. 1.]
print("Vt =\n", Vt)

# Reconstrucao: A = U diag(S) Vt
A_reconstruida = U @ np.diag(S) @ Vt
print("A reconstruida =\n", A_reconstruida)
```

A saída confirma os valores que calculamos à mão: determinante $3$, autovalores $3$ e $1$, valores singulares $3$ e $1$.

## Pseudocódigo: SVD simplificado (potência iterada)

O SVD pode ser aproximado pelo método da potência (para o maior valor singular):

```text
entrada: matriz A, número de iterações T
v = vetor aleatório normalizado
para t = 1 até T:
    v = A^T A v
    v = v / norma(v)          # normaliza
sigma = norma(A v)            # maior valor singular
devolver (sigma, v)
```

O vetor $v$ converge para o autovetor de $A^T A$ associado ao maior autovalor, e $\sigma$ é a raiz desse autovalor — o maior valor singular. Este é o mesmo princípio por trás da PCA e de decomposições de posto baixo.

## Erros comuns

- **Confundir transposta com inversa.** $A^T$ troca linhas por colunas; $A^{-1}$ desfaz a transformação. São coisas diferentes (só coincidem em matrizes ortogonais).
- **Achar que toda matriz tem inversa.** Matrizes singulares ($\det = 0$) não têm. Tentar inverter numericamente uma matriz quase singular produz resultados instáveis.
- **Esquecer que $(AB)^T = B^T A^T$.** A ordem inverte. Isso é crítico no backprop e em operações com atenção.
- **Interpretar autovalor negativo como "estica ao contrário".** Um autovalor negativo espelha a direção (vira o vetor para o lado oposto), além de escalar.
- **Confundir autovalores com valores singulares.** Valores singulares são sempre $\geq 0$ e vêm do SVD; autovalores podem ser negativos. Eles coincidem em módulo para matrizes simétricas.
- **Achar que o determinante grande significa transformação "boa".** Um determinante grande indica forte escala, mas também pode indicar má condicionamento (dimensões escaladas de forma muito desigual).

## Resumo

- A matriz **identidade** $I$ deixa todo vetor inalterado e é o neutro da multiplicação.
- A **transposta** $A^T$ espelha a matriz e guia o fluxo reverso do backprop ($W^T$).
- A **inversa** $A^{-1}$ desfaz a transformação, e só existe quando $\det(A) \neq 0$.
- O **determinante** mede o fator de escala de área/volume e detecta singularidade.
- **Autovalores e autovetores** descrevem as direções que a matriz apenas estica, e governam a estabilidade do treinamento.
- O **SVD** $A = U \Sigma V^T$ decompõe qualquer matriz em girar → esticar → girar, e fundamenta LoRA, PCA e compressão de pesos.
