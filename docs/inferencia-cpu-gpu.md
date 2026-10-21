# Inferência em CPU e GPU

Inferência é o uso de um modelo treinado para produzir previsões ou gerar saídas. CPU e GPU podem executá-la; a melhor escolha depende do modelo, da tarefa, do volume de solicitações, da memória disponível e do software usado.

## 1. O trabalho de inferência

No treino, o sistema calcula a saída, a perda e gradientes para atualizar os parâmetros. Na inferência, normalmente os parâmetros ficam fixos e não se calcula o backward. Isso reduz memória e trabalho, mas não torna toda inferência igualmente pequena.

Na geração autoregressiva há duas fases:

1. **Prefill:** o modelo processa o prompt para construir sua representação interna. Dependendo da arquitetura, isso pode envolver muitos tokens em paralelo.
2. **Decodificação:** o modelo produz um token e atualiza seu estado; a operação se repete até uma condição de parada.

Com cache KV em um Transformer, as chaves e os valores anteriores podem ser reutilizados, evitando recalcular todo o contexto em cada token. O cache ocupa memória que cresce com o comprimento do contexto e depende da arquitetura e das dimensões do modelo.

## 2. Métricas úteis

- **Latência até o primeiro token:** tempo entre o envio do prompt e a primeira saída. Inclui o prefill e outros custos do runtime.
- **Latência de decodificação:** tempo por token gerado, ou sua distribuição ao longo da resposta.
- **Throughput:** quantidade de tokens ou solicitações atendidas por unidade de tempo. Em um servidor com vários usuários, agregue a carga e a política de batching.
- **Memória:** inclui pesos, ativações temporárias, cache de contexto, buffers e o runtime.

Uma única medida de tokens/s não descreve toda a experiência. Registre o comprimento do prompt e da saída, o tamanho do lote, o hardware, a precisão e as opções de cache.

## 3. Como CPU e GPU diferem

Uma GPU pode processar muitas operações paralelas e costuma ser vantajosa em cargas grandes ou concorrentes, se o modelo e as operações couberem na memória e houver kernels adequados. A CPU pode ser conveniente quando não há GPU compatível, para cargas menores ou quando o modelo e o runtime estão otimizados para ela.

Não há uma regra universal de que a GPU sempre tenha menor latência ou de que a CPU seja suficiente para determinado tamanho de modelo. Transferências, inicialização, despacho de kernels, largura de banda de memória, paralelismo, quantização e concorrência mudam o resultado. Meça no hardware e na carga que importam.

| Fator | Pergunta a fazer |
|---|---|
| Memória | Pesos, cache e buffers cabem no dispositivo escolhido? |
| Latência | Quanto tempo leva o prefill e quanto leva cada token? |
| Concorrência | Quantas solicitações precisam ser atendidas ao mesmo tempo? |
| Runtime | Há kernels compatíveis e eficientes para o modelo e a precisão? |
| Custo | Qual é o consumo total de energia, tempo e recursos para a carga real? |

## 4. Memória e quantização

Como estimativa inicial, os pesos ocupam aproximadamente:

$$
\text{memória dos pesos} \approx \text{número de parâmetros} \times \text{bytes por parâmetro}
$$

Essa conta não inclui escalas, metadados, cache KV, ativações, buffers ou cópias temporárias; o uso real pode ser consideravelmente maior. A precisão de armazenamento e a precisão das operações também podem ser diferentes.

Quantização pode reduzir a memória dos pesos, mas a economia efetiva depende do formato e dos metadados. Ela só acelera a inferência quando o runtime possui caminhos eficientes para aquele formato. Pesos ternários podem permitir produtos simplificados, mas não garantem uma redução de latência ou energia em qualquer CPU ou GPU.

## 5. Executar um modelo

O código de inferência depende da biblioteca e do modelo. Em PyTorch, uma forma geral de preparar um modelo já carregado é:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

with torch.inference_mode():
    resultado = model(entrada.to(device))
```

`model.eval()` muda o comportamento de camadas como dropout; `torch.inference_mode()` evita registrar gradientes e pode reduzir sobrecarga. Eles têm funções distintas. A forma de obter logits, amostrar tokens e usar cache varia entre APIs de modelo, então consulte a documentação correspondente.

## 6. Fazer uma comparação justa

1. Use o mesmo modelo, entrada, saída desejada e configuração de geração.
2. Faça aquecimento antes de medir para separar custos de inicialização do ritmo estável.
3. Sincronize o dispositivo quando necessário; operações de GPU podem ser assíncronas.
4. Meça prefill e decodificação separadamente, além do tempo de carga se ele fizer parte do cenário real.
5. Repita a medição e registre latência, throughput, memória e condições do teste.

Exemplo de cronometragem de uma função que executa uma inferência:

```python
import time

def medir(funcao, sincronizar=None, repeticoes=10):
    for _ in range(2):
        funcao()  # aquecimento
    if sincronizar:
        sincronizar()

    inicio = time.perf_counter()
    for _ in range(repeticoes):
        funcao()
    if sincronizar:
        sincronizar()
    return (time.perf_counter() - inicio) / repeticoes
```

Para CUDA, passe uma função de sincronização apropriada, como `torch.cuda.synchronize`. A função medida deve reproduzir a carga de interesse e não misturar preparação de dados, a menos que esse custo também faça parte da comparação.

## Resumo

CPU e GPU são opções de execução, não uma classificação universal de velocidade. Prefill, decodificação, memória dos pesos, cache, concorrência e suporte do runtime afetam o resultado. Escolha com medições representativas e valide qualidade e custo junto com latência e throughput.
