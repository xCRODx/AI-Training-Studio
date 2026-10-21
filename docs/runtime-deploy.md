# Runtime e implantação de modelos

O runtime é a combinação de software e hardware que carrega um modelo e executa suas previsões. Escolher um formato de pesos ou um dispositivo não basta: o runtime precisa oferecer suporte à arquitetura, às operações, à precisão e ao método de quantização utilizados.

## 1. O ciclo de geração

Um modelo de linguagem autorregressivo gera uma sequência repetindo estes passos:

1. Converter o prompt em tokens.
2. Processar o contexto e calcular logits para o próximo token.
3. Aplicar a regra de seleção — por exemplo, greedy ou amostragem.
4. Acrescentar o token selecionado ao contexto e continuar até atingir um limite ou token de parada.
5. Converter os tokens produzidos de volta em texto.

O primeiro token pode exigir o processamento de todo o prompt. Depois disso, cada novo token usa o estado mantido pelo runtime, como um cache KV em Transformers ou um estado recorrente em outras arquiteturas.

## 2. CPU e GPU

**CPU:** adequada a muitos modelos pequenos, a cenários sem GPU e a runtimes otimizados com instruções vetoriais. A latência depende da quantidade de núcleos, largura de banda de memória, cache e paralelismo.

**GPU:** oferece grande paralelismo e pode acelerar operações matriciais, especialmente com lotes maiores. O resultado também depende da memória disponível, dos kernels, das transferências e do custo de iniciar cada operação.

Nenhum dispositivo é sempre mais rápido. Compare o modelo, a precisão, o tamanho do lote e o tipo de carga que realmente pretende executar.

## 3. Prefill e decode

- **Prefill:** processa o prompt inicial. Seu custo cresce com o comprimento do contexto e influencia a latência até a primeira resposta.
- **Decode:** gera tokens um por vez. O cache KV ou o estado recorrente influencia o custo e a memória ao longo da geração.

Por isso, throughput médio não descreve toda a experiência. Registre também a latência do primeiro token e a latência por token durante o decode.

## 4. Memória e estado

Durante a inferência, a memória pode incluir pesos, ativações temporárias, cache KV ou estado recorrente, buffers e dados do lote. O cache KV de um Transformer cresce com o contexto; estados recorrentes costumam manter tamanho fixo em relação ao número de tokens, mas podem exigir operações e dimensões internas próprias.

Quantização pode reduzir o espaço dos pesos, mas o ganho depende do formato e do runtime. A memória de contexto, os buffers e os tensores que não foram quantizados continuam contribuindo para o uso total.

## 5. Métricas para comparar runtimes

- **Latência do primeiro token:** tempo até o início da resposta.
- **Latência de decode:** tempo médio ou distribuição do tempo por token gerado.
- **Throughput:** tokens por segundo, indicando claramente se mede um usuário ou um lote de requisições.
- **Memória:** pico de RAM/VRAM e tamanho do cache ou estado.
- **Qualidade:** perplexidade, métricas da tarefa ou avaliação de saídas.

Faça medições repetidas com o mesmo modelo, prompt, tokenizador, parâmetros de amostragem, comprimento de contexto e hardware. Inclua aquecimento do runtime e diferencie tempo de carregamento de tempo de geração.

## 6. Formatos e compatibilidade

Formatos como safetensors e GGUF organizam pesos de maneiras diferentes. O primeiro é voltado a armazenar tensores; GGUF é usado por runtimes de inferência que suportam seus metadados e tipos. Nenhum formato garante compatibilidade universal. Consulte a documentação do runtime para verificar arquitetura, tokenizador e quantizações aceitos.

## 7. Erros comuns

- **Assumir que um arquivo válido será executável em qualquer runtime.** Formato e arquitetura precisam ser compatíveis.
- **Comparar apenas tokens por segundo.** Prefill, latência do primeiro token e uso de memória também importam.
- **Ignorar o tokenizador.** Um tokenizador incompatível produz IDs diferentes e invalida as entradas.
- **Compartilhar estado de geração entre sequências independentes.** Inicialize ou isole o cache e o estado por requisição.
- **Escolher hardware apenas pela especificação teórica.** Meça com a carga de trabalho de interesse.

## Resumo

- O runtime executa o ciclo de inferência e precisa implementar a arquitetura e o formato do modelo.
- CPU e GPU têm perfis distintos; o desempenho depende da carga e da implementação.
- Meça separadamente prefill, decode, latência, throughput, memória e qualidade.
- Formato de arquivo não garante suporte universal.
