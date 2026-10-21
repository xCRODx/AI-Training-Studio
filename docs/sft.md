# SFT (supervised fine-tuning)

## 1. A intuição: ensinar com exemplo e gabarito

Quando uma criança aprende a resolver equações, o professor não apenas entrega um livro de teoria e diz "leia". Ele **mostra** exemplos resolvidos: a equação de um lado, a resolução passo a passo do outro. A criança vê a entrada, vê a saída esperada, e aprende a relação entre elas. Se errar, recebe a correção imediata ("não é assim, é assim").

O **SFT — supervised fine-tuning, ou fine-tuning supervisionado** — faz exatamente isso com um modelo de linguagem. Ele apresenta **pares explícitos** de entrada e saída desejada e ajusta os pesos para que o modelo reproduza a saída correta dada a entrada.

> SFT é a forma de fine-tuning em que cada exemplo do conjunto de treino é um par rotulado `(entrada, saída)` — tipicamente `(pergunta, resposta)` ou `(instrução, resposta)`. O modelo aprende a gerar a saída correta minimizando uma perda supervisionada sobre a resposta.

A palavra **"supervisionado"** é a chave: existe um "gabarito". Diferentemente do pré-treino (onde o alvo é simplesmente o próximo token de um texto qualquer), no SFT um humano (ou um sistema que simula um humano) escreveu a resposta ideal e o modelo é explicitamente corrigido para se aproximar dela.

### Um exemplo concreto

| Entrada (prompt) | Saída (resposta esperada) |
|---|---|
| `Qual é a capital da França?` | `A capital da França é Paris.` |
| `Some 12 e 7.` | `12 + 7 = 19.` |
| `Traduza "olá" para inglês.` | `"Olá" em inglês é "hello".` |

O modelo vê a entrada e a saída juntas, e é treinado para, ao ver a entrada, produzir a saída. É o mecanismo que transforma um "modelo que completa texto" (que continuaria "Qual é a capital da França?" com qualquer coisa plausível) em um **assistente que responde perguntas**.

## 2. O conceito formal

### 2.1 A perda de next-token sobre a resposta

No SFT, a função de perda continua sendo a **entropia cruzada sobre o próximo token**, mas com uma diferença crucial: ela é calculada **apenas sobre os tokens da resposta**, não sobre os tokens da pergunta.

Dado um par $(x, y)$ onde $x$ é o prompt (pergunta/instrução) e $y$ é a resposta (sequência de $T_y$ tokens $y_1, y_2, \dots, y_{T_y}$), a perda por exemplo é:

$$
\mathcal{L}(x, y) = -\sum_{t=1}^{T_y} \log P\big(y_t \mid x, y_1, \dots, y_{t-1}\big)
$$

Expandindo a soma para um único exemplo:

$$
\mathcal{L}(x, y) = -\log P(y_1 \mid x) - \log P(y_2 \mid x, y_1) - \dots - \log P(y_{T_y} \mid x, y_{<T_y})
$$

A perda total sobre o dataset é a média:

$$
\mathcal{L}_{\text{SFT}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(x^{(i)}, y^{(i)})
$$

### 2.2 Por que mascarar o prompt?

Imagine que a perda fosse calculada sobre **todos** os tokens (prompt + resposta). Nesse caso, o modelo gastaria parte da sua capacidade aprendendo a "prever a pergunta" — o que é inútil, porque a pergunta é dada, não precisa ser prevista. Pior: em muitos modelos, isso dilui o sinal de aprendizado da resposta, que é o que realmente importa.

A solução técnica é a **máscara de perda**: atribuímos peso $0$ aos tokens do prompt e peso $1$ aos tokens da resposta. Com a máscara $m_t \in \{0, 1\}$:

$$
\mathcal{L}(x, y) = -\sum_{t=1}^{T} m_t \cdot \log P\big(y_t \mid \dots\big)
$$

onde $m_t = 1$ apenas para tokens que pertencem à resposta.

### 2.3 Exemplo numérico progressivo

Vamos acompanhar a perda token a token para o par `("Some 12 e 7.", "12 + 7 = 19.")`.

Suponha que o modelo, no início do treino, gere as seguintes probabilidades para os tokens-alvo da resposta:

| Posição | Token-alvo | $P(\text{alvo})$ | $-\log P$ | Mascarado? |
|---|---|---|---|---|
| 1 | `12` | 0.30 | 1.204 | Não (é prompt) |
| 2 | `+` | 0.50 | 0.693 | Sim |
| 3 | `7` | 0.40 | 0.916 | Sim |
| 4 | `=` | 0.60 | 0.511 | Sim |
| 5 | `19` | 0.20 | 1.609 | Sim |
| 6 | `.` | 0.80 | 0.223 | Sim |

A perda do exemplo é a soma apenas das linhas **mascaradas** (resposta):

$$
\mathcal{L} = 0.693 + 0.916 + 0.511 + 1.609 + 0.223 = 3.952
$$

O token que mais contribui é `19` ($1.609$), porque o modelo ainda não sabe o resultado da soma. À medida que o treino avança, a probabilidade de `19` sobe de $0.20$ para, digamos, $0.85$, e a contribuição desse token cai para $-\log(0.85) \approx 0.163$. A perda total do exemplo despenca:

$$
\mathcal{L}_{\text{após}} \approx 0.693 + 0.916 + 0.511 + 0.163 + 0.223 = 2.506
$$

O gradiente "empurra" justamente os tokens em que o modelo erra mais, guiado pelo gabarito supervisionado.

## 3. O formato de chat

O SFT moderno raramente usa pares soltos de texto. Em vez disso, usa um **formato de chat** estruturado, com papéis bem definidos.

### 3.1 Por que um formato?

Um modelo de linguagem precisa de **delimitadores** claros para saber onde termina a instrução e onde começa a resposta. Sem isso, ele não distingue "quem está falando". O formato de chat insere marcadores especiais (tokens ou strings) que separam os papéis.

### 3.2 Exemplo com o formato "user/assistant"

```text
<|im_start|>user
Qual é a capital da França?<|im_end|>
<|im_start|>assistant
A capital da França é Paris.<|im_end|>
```

Aqui, `<|im_start|>` e `<|im_end|>` são tokens especiais (ou marcadores textuais) que delimitam cada turno. Durante o treino:

1. Os tokens de `user` (a pergunta) entram no **contexto**, mas **não** contribuem para a perda.
2. Os tokens de `assistant` (a resposta) são os alvos supervisionados.

### 3.3 Tabela de marcadores comuns

| Sistema | Marcador de início | Papel do usuário | Papel do assistente |
|---|---|---|---|
| ChatML (OpenAI) | `<|im_start|>` | `user` | `assistant` |
| Llama 2 | `[INST]` / `[/INST]` | — | — |
| Llama 3 | `<|start_header_id|>` | `user` | `assistant` |
| Alpaca | `### Human:` / `### Assistant:` | texto | texto |

### 3.4 Multi-turno

Conversas com vários turnos são formatadas como uma sequência de mensagens, e a perda é calculada sobre **todas** as respostas do assistente, ignorando todos os prompts:

```text
<|im_start|>user
O que é BitNet?<|im_end|>
<|im_start|>assistant
BitNet é uma arquitetura que usa pesos ternários...<|im_end|>
<|im_start|>user
Quais as vantagens?<|im_end|>
<|im_start|>assistant
As vantagens incluem menor consumo de memória...<|im_end|>
```

O modelo aprende a responder o primeiro turno e, com o histórico no contexto, responder o segundo.

## 4. Código: construindo e treinando com SFT

### 4.1 Preparando os dados no formato de chat

```python
def formatar_sft(exemplo, tokenizador):
    """
    Converte um par pergunta/resposta no formato de chat
    e retorna ids de entrada e rótulos com máscara.
    """
    marcadores = tokenizador.marcadores_de_chat  # {'user': ..., 'assistant': ...}

    entrada = (
        marcadores["inicio"] + marcadores["user"] +
        exemplo["pergunta"] + marcadores["fim"] +
        marcadores["inicio"] + marcadores["assistant"] +
        exemplo["resposta"] + marcadores["fim"]
    )

    ids = tokenizador.encode(entrada)  # tokens de tudo

    # Máscara: 1 só para a parte da resposta
    ids_resposta = tokenizador.encode(
        marcadores["inicio"] + marcadores["assistant"] +
        exemplo["resposta"] + marcadores["fim"]
    )
    comprimento_resposta = len(ids_resposta)

    rotulos = [-100] * (len(ids) - comprimento_resposta) + ids[-comprimento_resposta:]

    return ids, rotulos
```

O truque do `-100` é o padrão do PyTorch para o `CrossEntropyLoss` com `ignore_index=-100`: qualquer posição com rótulo `-100` é ignorada no cálculo da perda. Assim, apenas os tokens da resposta geram gradiente.

### 4.2 Loop de treino completo


### 4.3 Pseudocódigo do processo

```text
ALGORITMO SFT(modelo_base, dataset_pares):
    PARA cada (pergunta, resposta) em dataset_pares:
        entrada = formata_chat(pergunta, resposta)
        ids = tokeniza(entrada)
        rotulos = [-100]*tamanho(pergunta) + ids_da_resposta
        logits = modelo(ids)
        perda = entropia_cruzada(logits, rotulos, ignorando -100)
        perda.retropropagar()
        otimizador.passo()
    RETORNA modelo_sft
```

## 5. Dados de exemplo

Um dataset SFT típico é um arquivo **JSONL** (uma linha por exemplo). Abaixo, três exemplos reais:

```json
{"pergunta": "Explique o que é bit em computação.", "resposta": "Um bit é a menor unidade de informação digital. Ele pode assumir dois valores: 0 ou 1."}
{"pergunta": "Qual é a fórmula da área de um círculo?", "resposta": "A área de um círculo é A = π r², onde r é o raio."}
{"pergunta": "Traduza 'bom dia' para o inglês.", "resposta": "'Bom dia' em inglês é 'good morning'."}
```

Esses pares são a matéria-prima do SFT. A qualidade deles determina, em grande medida, a qualidade do modelo final — detalhe explorado no texto sobre dataset de fine-tuning.

## 7. Erros comuns no SFT

### 7.1 Não mascarar o prompt

Se você esquece de aplicar a máscara, o modelo aprende a prever a pergunta. Sintoma: respostas que "repetem" a pergunta ou um treino que converge mais devagar. Solução: usar `ignore_index=-100` sobre os tokens do prompt.

### 7.2 Formato inconsistente entre exemplos

Se metade dos exemplos usa `### Human:` e a outra metade usa `<|im_start|>user`, o modelo fica confuso. Solução: padronizar o formato de chat em **todo** o dataset.

### 7.3 Respostas truncadas ou mal tokenizadas

Respostas que excedem o comprimento máximo são cortadas, gerando exemplos incompletos. Solução: filtrar ou truncar corretamente, e usar `attention_mask` para ignorar padding.

### 7.4 Desequilíbrio de comprimento

Muitas respostas curtas e poucas longas enviesam o modelo para respostas curtas. Solução: balancear ou controlar o comprimento.

### 7.5 Taxa de aprendizado inadequada

Alta demais destrói o conhecimento do pré-treino; baixa demais não ajusta nada. Solução: `lr` entre $10^{-5}$ e $10^{-4}$ e monitorar a curva de perda.

## 8. Resumo

1. **SFT** treina o modelo com pares rotulados `(entrada, saída)`.
2. A perda é a **entropia cruzada sobre o próximo token**, mas **apenas sobre os tokens da resposta**.
3. O **formato de chat** com marcadores delimita papéis e permite multi-turno.
4. A máscara (`-100`) impede que o modelo aprenda a prever o prompt.
5. O código consiste em: formatar, tokenizar, mascarar, prever, calcular perda, retropropagar.
7. Erros comuns: não mascarar, formato inconsistente, respostas truncadas e taxa inadequada.

A essência do SFT: **mostre ao modelo exatamente o que você quer que ele diga — e corrija-o até que ele diga.**
