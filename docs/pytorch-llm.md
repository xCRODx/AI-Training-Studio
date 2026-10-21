# PyTorch para LLMs

## 1. O que é o PyTorch e por que ele é usado em modelos de linguagem

O **PyTorch** é uma biblioteca de computação numérica que oferece tensores, operações para CPU e aceleradores compatíveis e um motor de diferenciação automática (*autograd*) para calcular gradientes.

Antes do PyTorch e de seus contemporâneos, treinar uma rede neural exigia derivar manualmente cada gradiente, implementar o *backward* na mão e, frequentemente, reescrever tudo quando a arquitetura mudava. O PyTorch inverte essa lógica: você escreve apenas o *forward* (a função que transforma a entrada na saída), e o autograd constrói o grafo computacional dinamicamente, permitindo que o *backward* seja calculado automaticamente com `loss.backward()`.

> **Intuição em uma frase:** você descreve o cálculo (*forward*) e, quando habilita o rastreamento, o PyTorch pode propagar gradientes por esse cálculo (*backward*); a escolha e a movimentação entre dispositivos continuam explícitas.


## 2. Tensores: a matéria-prima de tudo

### 2.1 O que é um tensor

Um **tensor** é a generalização de um número para dimensões arbitrárias:

- Um **escalar** é um tensor de ordem 0: `torch.tensor(3.14)`.
- Um **vetor** é um tensor de ordem 1: `torch.tensor([1.0, 2.0, 3.0])`.
- Uma **matriz** é um tensor de ordem 2: `torch.randn(4, 8)`.
- Um **lote de sequências** é um tensor de ordem 3: `torch.randn(batch, seq_len, d_model)`.

No contexto de um LLM, o tensor mais importante tem a forma `[batch, seq_len, d_model]`, que representa um lote de `batch` sequências, cada uma com `seq_len` tokens, cada token embebido em um vetor de `d_model` dimensões.

```python
import torch

batch, seq_len, d_model = 2, 512, 1024
x = torch.randn(batch, seq_len, d_model)
print(x.shape)   # torch.Size([2, 512, 1024])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

### 2.2 As três propriedades que importam

Todo tensor carrega três metadados fundamentais que decidem onde e como ele é computado:

| Propriedade | Significado | Exemplo |
|-------------|-------------|---------|
| `shape` | A estrutura dimensional | `[2, 512, 1024]` |
| `dtype` | O tipo numérico de cada elemento | `torch.float32`, `torch.bfloat16`, `torch.long` |
| `device` | O hardware onde o tensor vive | `cpu`, `cuda:0` |

Estes três são a raiz de praticamente todos os erros de principiante: um tensor em `cpu` não multiplica um tensor em `cuda`, e um tensor `int64` não soma com um `float32` sem conversão explícita.

### 2.3 Dtype importa (e muito) em LLMs

Os IDs de tokens são inteiros — `torch.long` (int64) —, enquanto embeddings, pesos e ativações costumam ser ponto flutuante. Em precisão mista, o tipo pode variar por operação e backend; a loss e algumas reduções podem usar precisão maior. Entender o `dtype` ajuda a diagnosticar incompatibilidades e erros numéricos.

```python
input_ids = torch.tensor([[101, 2023, 2003, 103]], dtype=torch.long)  # tokens: inteiros
emb = torch.randn(1, 4, 768, dtype=torch.bfloat16)                     # embeddings: bf16
```

### 2.4 Operações em lote

As operações sobre tensores são normalmente implementadas em código nativo e podem ser vetorizadas ou executadas em aceleradores compatíveis. Isso reduz a sobrecarga de laços Python, mas o desempenho depende da operação, do dispositivo e da implementação.

```python
a = torch.randn(1000, 1000)
b = torch.randn(1000, 1000)
c = a @ b          # matmul vetorizada: um único kernel CUDA, não 10^9 iterações
```

## 3. Autograd: o motor de diferenciação

### 3.1 O grafo computacional

Quando você cria um tensor com `requires_grad=True`, o PyTorch começa a registrar toda operação que o toca num **grafo computacional**. Cada nó guarda uma referência à função que o criou e às entradas usadas. No fim, `loss.backward()` percorre esse grafo de trás para frente aplicando a regra da cadeia.

```python
w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])
y = w * x          # y = 6.0
z = y ** 2         # z = 36.0
z.backward()       # dz/dw = 2 * y * x = 2 * 6 * 3 = 36
print(w.grad)      # tensor([36.])
```

A cadeia é $\frac{\partial z}{\partial w} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial w} = (2y)(x) = 2 \cdot 6 \cdot 3 = 36$.

### 3.2 Onde o autograd se conecta ao treino

Um ciclo comum de treino contém estas etapas:

1. Zerar os gradientes (`optimizer.zero_grad()`).
2. Computar o forward e a loss.
3. Chamar `loss.backward()` para propagar os gradientes.
4. Dar um passo do otimizador (`optimizer.step()`).

O autograd é o responsável por **como** o passo 3 funciona; o otimizador é o responsável por **o que fazer** com os gradientes calculados.

### 3.3 `torch.no_grad()` e `detach()`

Durante a inferência ou a avaliação, você não quer construir o grafo nem guardar ativações. O `torch.no_grad()` desliga o registro de gradiente num bloco inteiro:

```python
with torch.no_grad():
    logits = model(x)   # não constrói grafo, economiza memória
```


## 4. `nn.Module`: o bloco de construção universal

### 4.1 A anatomia de um módulo

Quase tudo no PyTorch é um `nn.Module`: uma camada, um bloco, um modelo inteiro. Um módulo tem:

- **Parâmetros** (`nn.Parameter`): tensores que o otimizador deve atualizar.
- **Submódulos**: outros `nn.Module` aninhados.
- **Buffers**: tensores que acompanham o módulo, mas não são parâmetros otimizados (por exemplo, estatísticas fixas ou contadores).

```python
import math
import torch
import torch.nn as nn

class Camada(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) / math.sqrt(in_f))
        self.register_buffer("contador", torch.zeros(1))

    def forward(self, x):
        return x @ self.weight.t()
```

Dois mecanismos tornam o `nn.Module` poderoso:

1. **Registro automático**: qualquer `nn.Parameter` ou submódulo atribuído em `__init__` entra em `self.parameters()` e em `self.state_dict()` automaticamente.
2. **Propagação de operações**: `model.to(device)`, `model.train()` e `model.eval()` percorrem todos os submódulos recursivamente.

### 4.2 `nn.ModuleList` e a composição

`nn.ModuleList` registra uma coleção de submódulos para que apareçam em `parameters()`, `state_dict()` e operações como `.to(device)`. Diferentemente de uma lista Python comum, seus elementos são reconhecidos pelo PyTorch.

```python
self.layers = nn.ModuleList([
    Bloco(d_model, num_heads)
    for _ in range(num_camadas)
])
```

## 5. `optim`: os otimizadores

### 5.1 O que um otimizador faz

Um otimizador recebe os parâmetros, lê os gradientes em `p.grad` e atualiza os pesos segundo uma regra. O mais simples, o SGD, faz:

$$
w \leftarrow w - \eta \cdot \nabla_w L
$$


$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
w \leftarrow w - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda w
$$

### 5.2 O estado do otimizador

O otimizador guarda um estado interno por parâmetro (os momentos $m$ e $v$). Esse estado é gravado separadamente no checkpoint:

```python
torch.save(optimizer.state_dict(), "optimizer.pt")
```

Para retomar uma execução com o mesmo estado do otimizador, salve e restaure esse dicionário junto com os parâmetros do modelo e o progresso relevante. Sem ele, os momentos adaptativos são reiniciados; isso pode mudar a trajetória do treino.

## 6. DataLoader, geradores e o fluxo de dados

### 6.1 O problema do streaming

Quando os dados não cabem convenientemente na memória, um iterável pode produzir lotes sob demanda. Este exemplo é um esqueleto: adapte o identificador do dataset, a coluna de texto, o tokenizador e as regras de fronteira ao conjunto real.

```python
from datasets import load_dataset
import torch

def data_generator(dataset_id, tokenizer_obj, context_length, batch_size):
    dataset = load_dataset(
        dataset_id,
        split="train",
        streaming=True,
    )
    buffer, batch_buffer = [], []
    for item in dataset:
        text = item.get("text", "")
        if not text:
            continue
        tokens = tokenizer_obj.encode(text, add_special_tokens=False)
        if tokenizer_obj.eos_token_id is not None:
            tokens.append(tokenizer_obj.eos_token_id)
        buffer.extend(tokens)
        while len(buffer) >= context_length + 1:
            chunk = buffer[:context_length + 1]
            del buffer[:context_length + 1]
            batch_buffer.append(chunk)
            if len(batch_buffer) == batch_size:
                yield torch.tensor(batch_buffer, dtype=torch.long)
                batch_buffer = []
```

O exemplo pressupõe que `load_dataset` tenha sido importado. Um gerador evita construir uma lista com todos os lotes de uma vez; o último lote incompleto não é emitido. Streaming, cache e retomada têm limitações específicas da fonte e da biblioteca; veja [Datasets em streaming](./dataset-streaming.md).

### 6.2 `DataLoader` versus gerador

`torch.utils.data.DataLoader` agrupa exemplos de datasets compatíveis e oferece opções como embaralhamento, pré-busca e workers. Seu comportamento depende do tipo de dataset; não é limitado a dados já carregados inteiramente na memória.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(torch.randint(0, 32000, (1000, 512)))
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

for batch in loader:
    ...
```


### 6.3 O shift causal

Em previsão autorregressiva, o alvo de cada posição é o token seguinte. Por isso, entrada e alvo são fatias deslocadas da mesma sequência:

```python
x = batch[:, :-1].to(device)           # entrada: tokens 0..N-1
y = batch[:, 1:].to(device)            # alvo: tokens 1..N
```

Essa é a essência do *causal language modeling*: cada posição prevê o próximo token. A forma exata da loss e o tratamento de padding dependem da tarefa.

## 7. Device e CUDA: movendo números para a GPU

### 7.1 A abstração de dispositivo

Um `torch.device` identifica onde tensores e parâmetros serão processados. O código pode escolher CPU ou CUDA em tempo de execução:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

Escolher CUDA não garante que o modelo caiba na memória nem que todas as operações tenham suporte nesse hardware.

### 7.2 As operações de movimento

| Operação | Efeito |
|----------|--------|
| `tensor.to(device)` | Move (copia) um tensor para o dispositivo |
| `model.to(device)` | Move todos os parâmetros e buffers do modelo |
| `tensor.cuda()` | Atalho para `.to("cuda")` |
| `tensor.cpu()` | Move de volta para a CPU |

A regra de ouro: **todos** os tensores que participam de uma operação devem estar no mesmo dispositivo. Misturar `cpu` e `cuda` numa matmul gera o clássico erro `Expected all tensors to be on the same device`.

## 8. AMP: precisão mista no PyTorch

AMP pode executar operações compatíveis em precisão reduzida. O suporte e o tipo adequado dependem do dispositivo e do modelo; teste estabilidade e qualidade antes de usar em uma execução longa.

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)
    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
```

Este exemplo pressupõe suporte a `bfloat16` em CUDA. Para outros tipos ou dispositivos, consulte [Precisão mista](./mixed-precision.md).

## 9. Um laço de treino completo e comentado

```python
import torch
import torch.nn as nn

def train_step(model, optimizer, batch, device, vocab_size):
    x = batch[:, :-1].to(device)
    y = batch[:, 1:].to(device)

    criterion = nn.CrossEntropyLoss()

    # 1. Forward em precisão padrão; AMP é opcional
    logits = model(x)
    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

    # Se houver padding, aplique uma máscara adequada à tarefa.
    # 2. Backward: autograd calcula os gradientes
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 3. Clip para estabilidade
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 4. Passo do otimizador e zeragem dos gradientes
    optimizer.step()

    return loss.item()
```

O exemplo mostra forward, loss, backward, clipping opcional e atualização. Se for usar acumulação de gradientes, a frequência de `zero_grad`, clipping e `optimizer.step()` deve ser ajustada ao ciclo de acumulação.

## 10. Tabela de API essencial

| API | Área | Para que serve | Exemplo de uso |
|---|---|---|---|
| `torch.tensor(...)` | Tensores | Cria um tensor a partir de dados | IDs de tokens ou métricas |
| `torch.randn(shape)` | Tensores | Gera valores aleatórios | Experimentos e inicialização |
| `torch.arange(n)` | Tensores | Cria uma sequência de inteiros | Índices de posição |
| `tensor.to(device, dtype)` | Tensores | Move ou converte um tensor | Dispositivo e tipo numérico |
| `tensor.reshape(...)` / `.view(...)` | Tensores | Reorganiza dimensões | Formatar logits para a loss |
| `torch.matmul(a, b)` / `@` | Álgebra | Multiplica matrizes | Projeções lineares |
| `tensor.mean/abs/sum` | Redução | Calcula estatísticas | Médias e normas |
| `torch.round/clamp` | Elementos | Arredonda ou limita valores | Operações de quantização |
| `nn.Module` | Módulos | Classe base para componentes | Camada ou modelo |
| `nn.Parameter` | Módulos | Registra um parâmetro otimizável | Pesos aprendidos |
| `nn.Linear` | Módulos | Aplica uma transformação linear | Projeção densa |
| `nn.ModuleList` | Módulos | Registra uma coleção de módulos | Pilha de blocos |
| `nn.CrossEntropyLoss` | Loss | Calcula entropia cruzada | Classificação ou próximo token |
| `optimizer.step()` | Otimização | Atualiza os parâmetros | Laço de treino |
| `optimizer.zero_grad()` | Otimização | Limpa gradientes acumulados | Início do ciclo de atualização |
| `loss.backward()` | Autograd | Propaga gradientes | Laço de treino |
| `torch.no_grad()` | Autograd | Suspende rastreamento de gradientes | Avaliação |
| `torch.autocast` | AMP | Seleciona precisão por operação | Cálculo compatível com o backend |
| `clip_grad_norm_` | Estabilidade | Limita norma global do gradiente | Antes do passo do otimizador |
| `torch.cuda.empty_cache()` | CUDA | Libera blocos não usados do cache do allocator | Uso pontual; não libera tensores ativos |
| `torch.save` / `torch.load` | Persistência | Salva ou carrega dados | Estado de treino |
| `DataLoader` | Dados | Itera e agrupa exemplos | Dataset compatível com indexação |

## 11. Erros comuns e como evitá-los

| Erro | Sintoma | Correção |
|------|---------|----------|
| Misturar `cpu` e `cuda` | `Expected all tensors to be on the same device` | Mova tudo com `.to(device)` antes da operação |
| Esquecer `zero_grad()` quando não se deseja acumulação | Gradientes de ciclos diferentes se somam | Zere no início do ciclo de atualização |
| Usar `optimizer.step()` antes do `backward()` | `grad is None` ou erro de atributo | Ordem: forward → backward → step |
| Esquecer `model.train()`/`eval()` | Dropout/normalização se comportam errado na inferência | Alterne os modos explicitamente |
| `view` em tensor não contíguo | `RuntimeError: view size is not compatible` | Use `.reshape(...)` ou `.contiguous().view(...)` |
| Misturar dtypes | `expected scalar type ... but found ...` | Converta com `.to(dtype)` |
| `requires_grad` em tensores de métricas | Grafo cresce e consome VRAM | Use `torch.no_grad()` em avaliação |
| Salvar só o modelo, não o otimizador | Retomada instável | Salve `optimizer.state_dict()` junto |
| Aumentar workers sem medir | Mais memória ou sobrecarga, às vezes sem ganho | Meça e confira o suporte do dataset/runtime |
| Ignorar `device` do dataset | Tensores nascem em CPU | `.to(device)` nos dados do batch |

## 12. Resumo

1. O PyTorch fornece tensores, operações numéricas e autograd; aceleração depende do dispositivo e do backend.
2. Um tensor tem propriedades importantes como `shape`, `dtype` e `device`.
3. O autograd constrói um grafo dinâmico; `loss.backward()` propaga gradientes pela regra da cadeia.
4. `nn.Module` registra parâmetros e submódulos, permitindo `state_dict()` e `.to(device)`.
5. `nn.Parameter` é otimizado; um buffer registrado acompanha o módulo sem ser atualizado pelo otimizador.
6. `DataLoader` e geradores atendem a diferentes formatos e escalas de dados; escolha conforme as necessidades.
7. No treino causal, entrada e alvo são fatias deslocadas, com tratamento adequado de padding e fronteiras.
8. AMP pode executar operações selecionadas em precisão reduzida; valide suporte, estabilidade e qualidade.
9. Um ciclo típico inclui forward, loss, backward e atualização; clipping e acumulação são opcionais conforme o treino.
