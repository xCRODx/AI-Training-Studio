# Precisão numérica na inferência

Reduzir a precisão dos pesos pode diminuir o espaço de armazenamento e, em plataformas compatíveis, acelerar a inferência. O resultado depende do modelo, do hardware e do runtime; usar menos bits não garante automaticamente menor latência nem qualidade equivalente.

## 1. O que muda ao usar 16 bits

Um peso em `float32` ocupa quatro bytes; em `bfloat16` ou `float16`, ocupa dois. Se todos os parâmetros forem armazenados nesses tipos, o payload dos pesos fica aproximadamente pela metade, sem contar metadados, buffers e outros tensores.

Isso pode reduzir o uso de memória ou o tráfego de dados. A velocidade só melhora se o hardware e os kernels executarem eficientemente esse formato e se o gargalo da carga for compatível com a redução.

## 2. `bfloat16` e `float16`

Os formatos usam a mesma quantidade total de bits, mas os distribuem de maneira diferente:

| Formato | Expoente | Fração explícita | Característica principal |
|---|---:|---:|---|
| `float32` | 8 bits | 23 bits | Mais precisão relativa |
| `bfloat16` | 8 bits | 7 bits | Faixa dinâmica semelhante à de `float32` |
| `float16` | 5 bits | 10 bits | Mais precisão relativa que `bfloat16`, faixa menor |

`float16` representa valores finitos até aproximadamente $65\,504$. `bfloat16` alcança uma faixa de magnitude próxima à de `float32`, mas representa menos dígitos significativos. Ambos arredondam valores que não são exatamente representáveis.

## 3. Escolher um formato

A escolha depende do suporte e do desempenho da plataforma. Algumas GPUs oferecem operações nativas para um ou ambos os formatos; CPUs e runtimes variam. Um formato que economiza memória pode ser mais lento se precisar de conversões ou não tiver kernels otimizados.

Valide a escolha com o modelo e a carga real. Observe memória, latência, throughput e métricas de qualidade. Não presuma que o modelo precise ter sido treinado exatamente no mesmo formato usado na inferência, nem que dois tipos de 16 bits produzam saídas idênticas.

## 4. Exemplo de inferência

O exemplo abaixo supõe que o dispositivo e o formato tenham sido escolhidos e validados para a plataforma:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16  # use apenas se o modelo e a plataforma oferecerem suporte

model = model.to(device=device, dtype=dtype).eval()
input_ids = input_ids.to(device)  # IDs permanecem inteiros

with torch.inference_mode():
    logits = model(input_ids)
    next_token_logits = logits[:, -1, :].float()
    probabilities = torch.softmax(next_token_logits, dim=-1)
```

Converter logits para `float32` antes do `softmax` é uma opção comum para fazer esse cálculo sensível em maior precisão. O formato apropriado pode variar conforme o runtime e a tarefa.

## 5. Erro numérico e qualidade

Converter pesos de maior precisão para 16 bits arredonda valores. Os erros podem afetar ativações e logits; não é correto presumir que sempre se cancelem ou que sejam imperceptíveis. Pequenas diferenças também podem alterar a ordem de logits muito próximos e, com amostragem, mudar o texto gerado.

Avalie o modelo convertido com dados representativos. Compare métricas e exemplos de saída com uma referência de maior precisão. Para avaliações rigorosas, mantenha o protocolo e a precisão documentados; igualdade bit a bit entre formatos diferentes não é uma expectativa adequada.

## 6. Quantização não é o mesmo que `bfloat16`

`bfloat16` é um formato de ponto flutuante de 16 bits. Quantização pode usar inteiros de baixa precisão, grupos com escalas ou representações especializadas, como pesos ternários. Esses métodos têm regras de escala, arredondamento e execução próprias.

Pesos ternários podem ser codificados com dois bits por valor, mas o modelo ainda pode precisar armazenar escalas, embeddings, normalizações e outros tensores em formatos diferentes. A precisão mista e a quantização podem coexistir em tensores distintos; o tamanho final depende da representação completa.

## 7. Erros comuns

- **Presumir que metade dos bytes significa o dobro da velocidade.** Meça no runtime e no hardware pretendidos.
- **Tratar `bfloat16` e `float16` como equivalentes.** Eles têm faixas dinâmicas e precisões diferentes.
- **Assumir que a qualidade não muda.** Valide métricas e saídas após converter o modelo.
- **Converter IDs de token para ponto flutuante.** IDs normalmente permanecem inteiros; o dtype se aplica aos cálculos e pesos.
- **Confundir ponto flutuante de 16 bits com quantização ternária.** São representações diferentes.
- **Esperar saídas idênticas entre formatos.** Arredondamento e amostragem podem alterar os resultados.

## Resumo

- `bfloat16` e `float16` usam 16 bits, mas distribuem expoente e fração de maneiras diferentes.
- Menor precisão pode reduzir memória; o efeito na velocidade depende do hardware e do runtime.
- Verifique qualidade e desempenho após alterar o formato.
- Quantização ternária e ponto flutuante de 16 bits são técnicas distintas, que podem ser usadas em partes diferentes do modelo.
