# Inferência eficiente em C++

Treinar e executar um modelo são tarefas diferentes. O treinamento precisa calcular gradientes e atualizar parâmetros; na inferência, os parâmetros ficam fixos e o modelo calcula previsões. Por isso, alguns sistemas treinam com ferramentas flexíveis e executam o modelo em um runtime otimizado, que pode ser escrito em C++.

## 1. O papel do runtime

Um runtime de inferência precisa interpretar a arquitetura, carregar pesos e metadados, processar entradas e produzir saídas. Um executável C++ não consegue executar qualquer modelo apenas por receber os pesos: ele precisa implementar as operações, formatos e regras de tokenização compatíveis com aquele modelo.

C++ pode oferecer controle sobre memória, paralelismo e instruções do processador. Isso não garante que seja mais rápido que Python: bibliotecas Python frequentemente delegam os cálculos a kernels altamente otimizados em C++ ou CUDA. O desempenho depende do runtime completo, do hardware e do tamanho do lote.

## 2. Aritmética com pesos ternários

Considere pesos quantizados $\widetilde{w}_j\in\{-1,0,+1\}$ e entradas $x_j$. O produto interno pode ser escrito como:

$$
y = \sum_{j: \widetilde{w}_j=+1}x_j - \sum_{j: \widetilde{w}_j=-1}x_j.
$$

Em uma implementação escalar, isso equivale a somar, subtrair ou ignorar cada entrada. Uma escala associada aos pesos, quando usada pelo método de quantização, também precisa ser aplicada ao resultado.

```cpp
float ternary_dot(const float* input, const signed char* weight, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        if (weight[i] > 0) sum += input[i];
        else if (weight[i] < 0) sum -= input[i];
    }
    return sum;
}
```

Esse exemplo demonstra a operação, não uma implementação de produção. Um laço condicional pode ser lento, e algumas instruções vetorizadas ainda executam multiplicações. O ganho pode vir da redução do tráfego de memória, da compactação, de operações inteiras ou de kernels especializados — e precisa ser medido.

## 3. Representação compacta

Três valores distintos podem ser codificados usando dois bits por peso, embora quatro combinações de bits estejam disponíveis. Quatro pesos podem, portanto, ocupar um byte. Esse é um formato possível; a ordem dos bits e o significado dos códigos precisam ser definidos pelo formato de arquivo e pelo runtime.

Dois bits por peso são uma contagem teórica para os pesos codificados. O modelo completo também pode incluir escalas, embeddings, normalizações, vocabulário e metadados em outras precisões. O tamanho total não é necessariamente reduzido na mesma proporção.

## 4. SIMD, GPU e paralelismo

Instruções SIMD permitem operar sobre vários valores em uma instrução, quando o algoritmo e o layout dos dados são compatíveis. GPUs oferecem paralelismo em maior escala, mas podem ter custos de transferência, lançamento de kernels e sincronização.

O paralelismo entre linhas ou blocos pode ajudar, mas adicionar threads não garante menor latência. Threads em excesso competem por núcleos, memória e cache. Compare tempos em condições controladas e registre o hardware, o tamanho do lote, o comprimento da sequência e a configuração do runtime.

## 5. Estado da geração

O custo de gerar texto depende do mecanismo da arquitetura. Um Transformer autorregressivo normalmente reutiliza um cache KV que cresce com o contexto. Arquiteturas recorrentes podem manter um estado de tamanho fixo em relação ao comprimento da sequência, embora o tamanho desse estado e o custo por token dependam da arquitetura.

Ao iniciar uma nova sequência independente, o runtime precisa inicializar o estado de geração correspondente. Compartilhar acidentalmente estado entre requisições pode misturar informações de entradas diferentes.

## 6. Como avaliar o desempenho

Separe pelo menos estas medidas:

- **Latência do primeiro token:** inclui o processamento do prompt (*prefill*).
- **Latência por token durante a geração:** ajuda a avaliar a fluidez do *decode*.
- **Throughput:** tokens produzidos por segundo em uma janela definida.
- **Uso de memória:** pesos, cache ou estado recorrente, buffers e lotes.

Compare a mesma arquitetura, precisão, entrada, configuração de amostragem e hardware. Validar a qualidade do modelo também é necessário: uma representação mais compacta pode alterar os resultados.

## 7. Erros comuns

- **Supor que C++ é automaticamente mais rápido.** O kernel, o runtime e o hardware determinam o desempenho.
- **Achar que um runtime aceita qualquer arquivo de modelo.** Arquitetura, nomes de tensores, tokenizador e quantização precisam ser compatíveis.
- **Contar apenas os bits dos pesos.** Escalas e outros tensores também ocupam espaço.
- **Inferir velocidade a partir da complexidade assintótica.** Meça prefill, decode, throughput e memória.
- **Presumir que pesos zero aceleram o cálculo.** É preciso que o kernel explore a esparsidade.

## Resumo

- Um runtime C++ precisa implementar explicitamente a arquitetura e os formatos que executa.
- Pesos ternários podem ser representados por dois bits e transformam o produto em somas com sinal, mas o ganho depende dos kernels e do hardware.
- SIMD, GPU e multithreading são ferramentas de otimização, não garantias de velocidade.
- Avalie prefill, geração, throughput, memória e qualidade de forma separada.
