# O que é fine-tuning

## 1. A intuição: do genérico ao específico

Imagine que você contratou uma pessoa extremamente culta e leiga em um assunto muito específico. Ela sabe de tudo: história, ciência, filosofia, literatura. Mas, ao receber uma pergunta sobre a rotina interna de uma empresa de logística — quais são os códigos de cada tipo de remessa, como se escreve um chamado de suporte, quais siglas os funcionários usam — ela não faz a menor ideia. Todo o conhecimento geral que ela tem não resolve o problema concreto do dia a dia dessa empresa.

O que você faz? Você não a manda de volta para a escola. Você senta com ela, mostra exemplos reais, explica o vocabulário interno, dá alguns casos resolvidos e pede para ela estudar aquele domínio. Depois de alguns dias de contato com esses exemplos, ela passa a responder com naturalidade usando os termos certos e o formato certo.

Isso é, em essência, **fine-tuning**. O modelo de linguagem pré-treinado (o "funcionário culto") já leu bilhões de palavras e domina a estrutura da língua, a lógica e uma enorme quantidade de fatos. Mas ele não foi treinado especificamente para a *sua* tarefa. O fine-tuning é o processo de **ajustar os pesos desse modelo** usando um conjunto de exemplos específicos, para que ele se comporte da maneira desejada — com o tom certo, o formato certo e o conhecimento certo para aquele domínio.

> Fine-tuning é o reajuste de um modelo já treinado, com dados novos e específicos, para especializá-lo em uma tarefa. Ele aproveita tudo o que o modelo já sabe e apenas "desloca" esse conhecimento na direção do que você precisa.

### Por que não treinar do zero?

Treinar um modelo do zero custa caro. Um modelo de alguns bilhões de parâmetros exige milhares de GPUs, semanas de processamento e terabytes de dados. O fine-tuning, por outro lado, pode ser feito com **uma fração minúscula** desse custo, porque o modelo já "sabe falar". Você só está ensinando um novo registro, uma nova tarefa, uma nova formatação.

Para fixar a intuição, pense em três níveis de especialização:

| Nível | Descrição | Custo | Exemplo |
|---|---|---|---|
| Pré-treino | Aprender a língua do zero, sobre textos gigantes | Altíssimo | "Aprender a falar e escrever" |
| Fine-tuning | Ajustar para uma tarefa/domínio com exemplos | Médio/baixo | "Aprender o jargão de suporte técnico" |
| Inferência | Usar o modelo finalizado | Mínimo | "Responder ao cliente" |

## 2. O conceito formal

### 2.1 O que significa "ajustar os pesos"

Um modelo de linguagem é, na prática, uma função gigante parametrizada por pesos $\theta$. Dado um texto de entrada $x$, ele produz uma distribuição de probabilidade sobre o próximo token:

$$
P(y_t \mid x, y_{<t}) = \text{softmax}\big(f_\theta(x, y_{<t})\big)
$$

Durante o pré-treino, os pesos $\theta$ são otimizados para **maximizar a verossimilhança** de prever o próximo token em uma coleção massiva de texto. Chamamos esses pesos de $\theta_{\text{pretrain}}$.

No fine-tuning, partimos de $\theta_{\text{pretrain}}$ (não de pesos aleatórios!) e continuamos o treinamento, agora com uma função de perda específica para a tarefa, sobre um conjunto de dados $D_{\text{ft}}$ muito menor:

$$
\theta_{\text{ft}} = \arg\min_{\theta} \; \mathcal{L}\big(f_\theta, D_{\text{ft}}\big), \quad \text{iniciando de } \theta_{\text{pretrain}}
$$

A palavra-chave é **"iniciando de"**. O modelo não parte do zero: ele parte de um ótimo já encontrado. Isso faz com que o fine-tuning convirja em pouquíssimas épocas, com poucos exemplos.

### 2.2 A matemática da perda

Para tarefas de linguagem, a perda mais comum continua sendo a **entropia cruzada** sobre o próximo token, a mesma usada no pré-treino:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P(y^{(i)}_t \mid x^{(i)}, y^{(i)}_{<t})
$$

Onde:

- $N$ é o número de exemplos do dataset de fine-tuning.
- $T_i$ é o comprimento (em tokens) do exemplo $i$.
- $y^{(i)}_t$ é o token-alvo na posição $t$.

A diferença em relação ao pré-treino não está na fórmula, mas **nos dados**: em vez de texto genérico da internet, usamos pares estruturados de entrada-saída (pergunta/resposta, instrução/resposta, texto/resumo etc.).

### 2.3 Um exemplo numérico progressivo

Vamos construir um entendimento numérico do que acontece. Suponha um vocabulário minúsculo com 5 tokens: `{A, B, C, D, <fim>}`.

**Antes do fine-tuning**, dado o prefixo "Qual é a capital da França?", o modelo pré-treinado poderia distribuir probabilidade assim (valores ilustrativos):

$$
P_{\text{pretrain}} = [0.10,\ 0.08,\ 0.70,\ 0.05,\ 0.07]
$$

onde `C` (digamos, a palavra "Paris") já tem $70\%$ de chance — o modelo *sabe*, mas não com a confiança e o formato que desejamos.

**Após o fine-tuning** com exemplos do tipo `{"pergunta": "Qual é a capital da França?", "resposta": "Paris"}` repetidos de forma consistente, a distribuição se "afunila":

$$
P_{\text{ft}} = [0.01,\ 0.01,\ 0.96,\ 0.01,\ 0.01]
$$

A massa de probabilidade se concentrou no token correto. Em termos de entropia cruzada, a perda caiu de aproximadamente $-\log(0.70) \approx 0.357$ para $-\log(0.96) \approx 0.041$. O modelo ficou **mais certo** sobre o que deve responder.

## 3. Pré-treino vs fine-tuning: o contraste

Entender a diferença entre essas duas etapas é o coração deste texto. Vamos comparar em detalhes.

### 3.1 O pré-treino

No pré-treino, o objetivo é **aprender a língua e o mundo**. O modelo recebe enormes quantidades de texto (livros, Wikipedia, código, sites) e aprende a prever a próxima palavra. Ele desenvolve:

- **Sintaxe e gramática**: sujeito-verbo-objeto, concordância, pontuação.
- **Conhecimento factual**: quem foi uma figura histórica, fórmulas, fatos.
- **Raciocínio básico**: implicações lógicas simples, padrões.
- **Estilo e registro**: como se escreve de forma formal, informal, técnica.

O pré-treino é **não supervisionado** no sentido de que os rótulos são o próprio texto: cada token seguinte é o "alvo" do token anterior. Não há um humano criando pares pergunta-resposta.

### 3.2 O fine-tuning

No fine-tuning, o objetivo é **aprender um comportamento**. O modelo recebe exemplos curados que mostram *como* ele deve responder: o formato, o tom, o estilo, e o conhecimento específico do domínio. Ele aprende:

- **Formato de resposta**: responder em uma linha, em JSON, em Markdown.
- **Comportamento conversacional**: ser útil, educado, objetivo.
- **Conhecimento de domínio**: siglas, regras, procedimentos internos.
- **Alinhamento**: não dar respostas tóxicas, seguir instruções.

### 3.3 Tabela comparativa

| Dimensão | Pré-treino | Fine-tuning |
|---|---|---|
| Objetivo | Aprender a língua e o mundo | Aprender um comportamento/tarefa |
| Dados | Texto genérico massivo (TB) | Pares curados (MB a GB) |
| Rótulos | Auto-supervisionado (próximo token) | Supervisionado (humano ou sintético) |
| Custo computacional | Altíssimo (semanas, muitas GPUs) | Médio/baixo (horas) |
| Peso inicial | Aleatório | Pesos do pré-treino |
| Convergência | Lenta, muitas épocas | Rápida, poucas épocas |
| O que muda | Tudo | Um "deslocamento" fino |

## 4. Tipos de fine-tuning

Nem todo fine-tuning é igual. Há uma família de variações:

### 4.1 Fine-tuning completo (full fine-tuning)

Todos os pesos $\theta$ do modelo são atualizados. É o mais poderoso, mas exige mais memória e pode causar **esquecimento catastrófico** (o modelo "desaprende" conhecimento geral).

### 4.2 Fine-tuning parcial

Apenas algumas camadas são atualizadas (por exemplo, só as últimas). As camadas iniciais, que capturam características genéricas, ficam congeladas.

### 4.3 Fine-tuning eficiente em parâmetros (PEFT)

Em vez de atualizar os pesos originais, adicionamos um **número pequeno de parâmetros novos** e treinamos apenas eles. A técnica mais conhecida é a **LoRA**, que representa a atualização $\Delta W$ da matriz de pesos por meio de fatores de posto baixo (*low-rank*):

$$
W_{\text{novo}} = W_0 + \Delta W, \quad \Delta W = B A
$$

onde $W_0 \in \mathbb{R}^{d \times k}$ é a matriz congelada, e $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ são as matrizes treináveis de baixa dimensão ($r \ll d, k$).

### 4.4 SFT (supervised fine-tuning)


## 5. Código: esqueleto de um loop de fine-tuning

Para tornar a ideia concreta, vejamos um esqueleto (simplificado, mas fiel à lógica) de um loop de fine-tuning com PyTorch.


Os pontos-chave do código:

1. **Taxa de aprendizado baixa** (`lr=2e-5`): não queremos destruir o conhecimento pré-treinado.
2. **Poucas épocas** (3): fine-tuning converge rápido.
3. **`ignore_index=-100`**: tokens de *prompt* (pergunta) não contribuem para a perda — só a resposta conta (detalhe que será aprofundado no texto sobre SFT).
4. **Deslocamento de 1 token**: o modelo prevê o token $t+1$ a partir do token $t$.

### Pseudocódigo de alto nível

```text
ALGORITMO FineTuning(modelo, dataset, epocas, lr):
    inicializar otimizador com lr baixo
    PARA cada época:
        PARA cada (entrada, alvo) em dataset:
            logits = modelo(entrada)
            perda = entropia_cruzada(logits, alvo, ignorando_prompt)
            perda.retropropagar()
            otimizador.passo()
    RETORNAR modelo_ajustado
```

## 7. Erros comuns no fine-tuning

### 7.1 Taxa de aprendizado alta demais

Como o modelo já está em um bom ponto, uma taxa alta "empurra" os pesos para longe e destrói o conhecimento geral. Sintoma: a perda cai rápido e depois **diverge** (sobe, vira `NaN`). Solução: começar com valores como $10^{-5}$ a $10^{-4}$.

### 7.2 Muitas épocas

Treinar demais leva ao **overfitting** e ao **esquecimento catastrófico**. O modelo "decora" os exemplos e perde generalização. Solução: monitorar a perda em um conjunto de validação e usar *early stopping*.

### 7.3 Dados desbalanceados ou em formato errado

Se o dataset mistura estilos (JSON, texto livre, tabelas) sem consistência, o modelo aprende um "ruído" de formato. Solução: curar o dataset (veja o texto sobre dataset de fine-tuning).

### 7.4 Esquecer de mascarar o prompt

Se a perda é calculada também sobre os tokens da pergunta, o modelo gasta capacidade aprendendo a "prever a pergunta" em vez de "responder". Solução: mascarar os tokens de entrada com `ignore_index=-100`.

### 7.5 Não validar com métricas da tarefa

Perda baixa não significa resposta boa. Solução: avaliar com métricas de tarefa (exatidão, F1, ou mesmo um *check* manual de qualidade).

## 8. Resumo

1. **Fine-tuning** é o reajuste de um modelo **pré-treinado** com dados específicos, partindo de $\theta_{\text{pretrain}}$.
2. Ele é barato porque aproveita o conhecimento já existente: poucos dados, poucas épocas, taxa de aprendizado baixa.
3. O pré-treino ensina a **língua**; o fine-tuning ensina o **comportamento**.
4. Existem variantes: full, parcial, PEFT/LoRA e SFT.
5. A perda típica é a **entropia cruzada sobre o próximo token**, mas restrita aos tokens da resposta.
7. Erros comuns incluem taxa alta, épocas demais, dados sujos e ausência de máscara no prompt.

A intuição central permanece: **você não ensina o modelo a falar de novo — você o ensina a falar do seu jeito.**
