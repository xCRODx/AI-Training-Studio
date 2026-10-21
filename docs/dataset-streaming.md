# Datasets em streaming: processar dados maiores que a memória

Um dataset pode ser maior que a RAM disponível. Em vez de carregar todos os exemplos antes de começar, é possível percorrê-los sob demanda. Essa estratégia é chamada de **streaming**. Dados grandes também podem estar divididos em arquivos menores, ou *shards*, mas a organização depende da fonte.

## 1. Carregamento integral e streaming

No carregamento integral, os dados são preparados para operações como acesso por índice, embaralhamento amplo e múltiplas passagens. Isso pode ser conveniente, mas usa espaço de armazenamento e memória para cache ou processamento.

No streaming, um iterador entrega exemplos conforme o consumidor os solicita:

```python
from datasets import load_dataset

stream = load_dataset(
    "organizacao/nome-do-dataset",
    split="train",
    streaming=True,
)

for exemplo in stream:
    print(exemplo)
    break
```

O identificador é ilustrativo. Confirme o esquema e a licença do dataset escolhido. Um iterador não deve ser tratado como uma lista: comprimento, indexação aleatória e embaralhamento global podem não estar disponíveis. O embaralhamento de um stream costuma usar um buffer, portanto não equivale necessariamente a permutar o conjunto inteiro.

## 2. Custo de memória

Se houver $N$ exemplos, cada um com em média $T$ tokens armazenados em inteiros de $b$ bytes, materializar todos os tokens exige aproximadamente:

$$
M_{\text{integral}} \approx N \cdot T \cdot b
$$

Por exemplo, $10^6$ exemplos com 512 tokens em inteiros de 8 bytes representam cerca de 4,1 GB, sem contar os demais dados e estruturas de processamento.

Com streaming, o consumo tende a depender dos buffers ativos, do lote atual, da pré-busca e do cache, não do total de exemplos. Portanto, streaming **reduz** a necessidade de materializar o corpus, mas não torna o uso de memória literalmente constante: bibliotecas, leitores, workers e caches também reservam memória.

## 3. Shards e leitura sob demanda

Uma fonte pode dividir os dados em arquivos ou shards. O leitor percorre esses fragmentos conforme a implementação e o formato permitem; não se deve presumir que todos os datasets sejam Parquet, que cada linha exija uma requisição de rede ou que o cursor corresponda sempre a um simples par arquivo/linha.

Sharding pode ajudar a distribuir o trabalho entre processos ou máquinas. Cada worker precisa receber uma partição distinta para evitar que todos processem os mesmos exemplos. A forma de particionar depende do dataset e do framework.

## 4. Tokenizar e formar sequências

Para treino causal, uma estratégia comum é concatenar IDs de tokens e dividi-los em sequências de comprimento $L+1$. Os primeiros $L$ tokens formam a entrada; os seguintes, deslocados em uma posição, são os alvos.

```python
def gerar_lotes(stream, tokenizer, comprimento, tamanho_lote):
    buffer_tokens = []
    sequencias = []

    for exemplo in stream:
        texto = exemplo.get("text", "")
        if not texto:
            continue

        ids = tokenizer.encode(texto, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        buffer_tokens.extend(ids)

        while len(buffer_tokens) >= comprimento + 1:
            sequencias.append(buffer_tokens[:comprimento + 1])
            del buffer_tokens[:comprimento + 1]

            if len(sequencias) == tamanho_lote:
                yield sequencias
                sequencias = []
```

Esse exemplo é didático: adapta-se a datasets com outra estrutura e a tarefas que não devam concatenar documentos. O token de fim de sequência ajuda a marcar fronteiras, mas a escolha correta depende do tokenizador e do formato de treino. O código descarta a sequência e o lote incompletos no fim do stream; outra aplicação pode optar por guardá-los ou completá-los.

Para um lote de forma $(B, L+1)$, o deslocamento causal fica assim:

```python
x = lote[:, :-1]  # tokens de entrada
y = lote[:, 1:]   # próximo token esperado
```

## 5. Retomada do iterador

Alguns iteráveis oferecem métodos como `state_dict()` e `load_state_dict()` para serializar o progresso. Quando existirem, eles podem ajudar a retomar a leitura:

```python
if hasattr(stream, "state_dict"):
    estado = stream.state_dict()
    salvar_estado(estado)

# Ao reconstruir um stream compatível:
if estado_salvo is not None and hasattr(stream, "load_state_dict"):
    stream.load_state_dict(estado_salvo)
```

O suporte e o nível de precisão da retomada dependem da biblioteca, do formato, da configuração de shuffle, dos workers e da versão. Salve também metadados suficientes para recriar a mesma fonte e configuração; teste a restauração antes de depender dela em treinos longos. Um estado do iterador não inclui automaticamente o estado do modelo, do otimizador ou de todos os buffers de pré-processamento.

Se não houver estado serializável, pular exemplos pode permitir retomar aproximadamente, mas pode ser lento e não garante a mesma posição em tokens. `skip(n)` nem sempre é um salto aleatório direto: o leitor pode precisar consumir os exemplos anteriores.

## 6. Throughput e pré-busca

Leitura remota ou de disco pode limitar a velocidade. Prefetch, buffers e leitura paralela podem sobrepor parte do trabalho de dados ao cálculo do modelo, desde que haja memória e suporte adequados. Eles também podem aumentar consumo de recursos ou complexidade.

Compare as alternativas medindo o tempo de preparação dos dados e o tempo total de treino em condições equivalentes. Streaming não é automaticamente mais lento nem tão rápido quanto o carregamento integral: rede, cache, formato, tokenização, workers e hardware fazem diferença.

## 7. Exemplo de divisão em sequências

Considere os tokens `[o, gato, comeu, o, rato]` e um comprimento de entrada $L=3$. O buffer fornece uma sequência de quatro tokens:

```text
entrada: [o, gato, comeu]
alvo:    [gato, comeu, o]
```

O token `rato` ainda pode participar de uma sequência posterior quando mais tokens chegarem. Em produção, fronteiras entre documentos, tokens especiais e exemplos incompletos precisam ser tratados de acordo com a tarefa.

## 8. Erros comuns

| Erro | Consequência | Boa prática |
|---|---|---|
| Usar indexação ou `len()` como se o stream fosse uma lista | Operação indisponível ou custo inesperado | Trabalhar com iteração e verificar o suporte da fonte |
| Presumir shuffle global | Ordem mais local ou enviesada | Entender o shuffle por buffer e usar seeds quando suportadas |
| Supor que o estado do iterador garante retomada exata | Repetição ou omissão de exemplos | Validar o mecanismo com a mesma fonte e configuração |
| Juntar documentos sem considerar fronteiras | Contextos atravessam exemplos indevidamente | Inserir separadores adequados ou manter documentos separados |
| Contar apenas o buffer de tokens na estimativa de memória | Subestimar uso real | Incluir lotes, pré-busca, workers e cache |
| Usar workers sem particionar o stream | Exemplos duplicados entre workers | Configurar sharding/distribuição conforme a biblioteca |

## Resumo

Streaming permite consumir dados sob demanda e evitar materializar todo o corpus. A memória passa a depender principalmente dos buffers e da infraestrutura de leitura, enquanto o custo de I/O e as limitações de iteração precisam ser considerados. Sharding e retomada são recursos dependentes da fonte e da biblioteca; valide-os antes de assumir que preservam exatamente a ordem ou a posição.
