# Hugging Face Datasets e Transformers

Hugging Face reúne ferramentas para descobrir e carregar datasets e para usar modelos e tokenizadores. As bibliotecas são independentes: `datasets` cuida dos dados; `transformers` fornece componentes de modelos e tokenização.

## 1. Carregar um dataset

O `load_dataset` pode carregar dados de um repositório público, de arquivos locais ou de um script compatível. Um dataset pode oferecer várias configurações e partições.

```python
from datasets import load_dataset

dataset = load_dataset("organizacao/nome-do-dataset", split="train")
print(dataset[0])
```

O identificador é apenas um exemplo: substitua-o por um dataset disponível e confira sua licença, descrição, colunas e condições de acesso antes de usá-lo.

| Argumento | Para que serve |
|---|---|
| `path` | Identifica o dataset ou a fonte de dados |
| `name` | Seleciona uma configuração, quando o dataset oferece mais de uma |
| `split` | Seleciona uma partição, como `train` ou `test` |
| `streaming` | Lê os exemplos sob demanda em vez de materializar a partição |
| `token` | Fornece autenticação quando a fonte exige acesso autorizado |

Nem todo dataset tem as mesmas colunas ou partições. Inspecione alguns registros e adapte o pré-processamento ao esquema real.

## 2. Transformar exemplos

`Dataset.map` aplica uma função aos exemplos. Pode ser usado para limpar texto, selecionar campos ou tokenizar os dados. A função deve refletir o formato do dataset, não presumir que toda linha contenha uma coluna chamada `text`.

```python
def normalizar_exemplo(exemplo):
    texto = exemplo["text"].strip()
    return {"text": texto}

dataset_limpo = dataset.map(normalizar_exemplo)
```

Para tokenizar texto com um tokenizador compatível:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nome-ou-identificador-do-tokenizador")

def tokenizar_lote(lote):
    return tokenizer(lote["text"], truncation=True)

dataset_tokenizado = dataset_limpo.map(tokenizar_lote, batched=True)
```

Use o tokenizador apropriado ao modelo ou à tarefa. `truncation=True` limita exemplos ao tamanho aceito pelo tokenizador; escolha esse limite de acordo com o contexto e com o objetivo, em vez de assumir um valor universal.

## 3. Carregamento integral e streaming

No modo padrão, `load_dataset` prepara uma partição para acesso repetido e pode usar cache local. Com `streaming=True`, retorna um iterável que entrega exemplos sob demanda:

```python
stream = load_dataset(
    "organizacao/nome-do-dataset",
    split="train",
    streaming=True,
)

for exemplo in stream:
    texto = exemplo.get("text", "")
    if texto:
        print(texto[:100])
        break
```

Streaming pode reduzir a memória necessária para começar a processar dados grandes, mas não elimina toda a memória de buffers, cache ou pré-busca. Algumas operações de acesso aleatório e de contagem também não estão disponíveis ou têm limitações. Veja [Datasets em streaming](./dataset-streaming.md) para os trade-offs.

## 4. Acesso restrito e credenciais

Alguns repositórios exigem autenticação ou aceitação de termos. Siga as condições informadas pelo publicador e use uma credencial armazenada com segurança; não a escreva no código, em notebooks compartilhados ou nos dados de saída.

```python
import os
from datasets import load_dataset

dataset = load_dataset(
    "organizacao/dataset-restrito",
    split="train",
    token=os.environ.get("HF_TOKEN"),
)
```

Se a fonte for pública, o argumento `token` normalmente pode ser omitido. Uma credencial só concede o acesso permitido à conta; ela não substitui a verificação da licença nem das regras de uso dos dados.

## 5. Boas práticas

- Leia o cartão do dataset e confirme licença, idioma, proveniência e possíveis dados sensíveis.
- Inspecione exemplos antes de escolher colunas, regras de limpeza e estratégia de tokenização.
- Separe treino, validação e teste sem deixar exemplos duplicados ou relacionados atravessarem as partições.
- Registre a versão ou revisão da fonte, o tokenizador e as transformações para facilitar a reprodução.
- Trate erros de rede, permissões e cache como condições possíveis; nem todo dataset está sempre disponível.

## Resumo

`datasets` ajuda a carregar e transformar dados; `transformers` oferece tokenizadores e modelos. O formato, o acesso e as operações disponíveis dependem da fonte escolhida. Inspecione os dados, use credenciais com segurança e escolha entre carregamento integral e streaming conforme o tamanho e as necessidades do processamento.
