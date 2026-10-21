# Conversão de modelos para GGUF

Converter um modelo para GGUF é transformar pesos e metadados em um formato de inferência que possa ser lido por runtimes compatíveis. A conversão pode preservar os pesos em ponto flutuante ou quantizá-los. Ela não é apenas uma mudança de extensão: arquitetura, nomes de tensores, tokenizador e tipos numéricos precisam corresponder ao que o runtime implementa.

## 1. Compatibilidade antes da conversão

Antes de converter, confirme:

- se o runtime de destino oferece suporte à arquitetura e às operações do modelo;
- quais tipos de tensor e métodos de quantização são aceitos;
- como os nomes e as dimensões dos tensores de origem correspondem ao formato de destino;
- se o tokenizador e os tokens especiais podem ser incluídos ou precisam ser distribuídos separadamente;
- quais metadados são necessários para reconstruir a configuração.

Um arquivo GGUF válido não implica que qualquer runtime consiga executar o modelo. O formato organiza dados e metadados; o runtime ainda precisa implementar a arquitetura.

## 2. Etapas conceituais

Um conversor costuma seguir este fluxo:

1. Ler e validar pesos e configuração de origem.
2. Mapear os nomes, dimensões e layouts dos tensores para a arquitetura de destino.
3. Opcionalmente converter a precisão ou quantizar tensores selecionados.
4. Escrever metadados, informações do tokenizador quando aplicável e dados dos tensores.
5. Validar o arquivo com um leitor compatível e executar uma inferência de teste.

O conjunto exato de tensores transformados depende do modelo e do método de quantização. Embeddings, normalizações, escalas e outros componentes podem permanecer em tipos numéricos diferentes dos usados nas matrizes lineares.

## 3. Exemplo de quantização ternária

Uma regra didática para uma matriz $W$ é:

$$
\alpha=\operatorname{mean}(|W|), \qquad
\widetilde{W}=\operatorname{clip}(\operatorname{round}(W/\alpha),-1,1).
$$

Por exemplo, $[0.003,-0.021,0.012,-0.0004]$ tem escala média $\alpha=0.0091$. Com essa regra, os valores quantizados são $[0,-1,+1,0]$. O conversor precisa armazenar a escala ou outra informação equivalente se o método exigir a reconstrução da magnitude.

Essa fórmula ilustra uma possível quantização ternária; não define todas as variantes BitNet. Escalas, arredondamento, agrupamento e tratamento de valores extremos variam entre métodos.

## 4. Empacotamento em bits

Como $-1$, $0$ e $+1$ são três estados, cada peso pode ser codificado com dois bits. Um esquema possível mapeia $-1\to00$, $0\to01$ e $+1\to10$, permitindo armazenar quatro pesos por byte.

O código e a ordem dos bits são convenções do formato adotado. Um conversor e um runtime precisam usar exatamente a mesma convenção, tratar corretamente dimensões não múltiplas do tamanho do bloco e manter as escalas associadas aos pesos.

Se $N$ pesos forem armazenados em dois bits cada, o payload ocupa idealmente $N/4$ bytes. Isso representa até 16 vezes menos bytes que os mesmos pesos em `float32`, mas não significa que o arquivo inteiro ou a memória do modelo ficarão 16 vezes menores: outros tensores e metadados permanecem, e alinhamento e cabeçalhos adicionam espaço.

## 5. Escrita e validação

Para evitar publicar um arquivo incompleto, conversores podem gravar em um destino temporário, validar o resultado e só então renomeá-lo para o nome final. Valide tanto a estrutura do arquivo quanto o comportamento do modelo:

- abra-o com um leitor que suporte a versão e os tipos utilizados;
- verifique nomes, dimensões e metadados esperados;
- execute entradas curtas conhecidas e confirme que a geração não falha;
- quando possível, compare logits, perplexidade ou métricas da tarefa com o modelo de origem;
- meça qualidade, memória e velocidade no runtime de destino.

Diferenças de precisão e amostragem podem produzir textos diferentes sem indicar corrupção. Compare saídas determinísticas ou métricas apropriadas quando precisar avaliar conversões.

## 6. Problemas frequentes

| Problema | Consequência | Verificação |
|---|---|---|
| Arquitetura não suportada pelo runtime | O arquivo pode ser lido, mas o modelo não executa | Consulte a matriz de compatibilidade do runtime |
| Tensor renomeado ou transposto incorretamente | Erro de dimensão ou previsões incorretas | Compare nomes, formas e convenções da arquitetura |
| Escala descartada ao quantizar | Magnitudes das saídas ficam incorretas | Preserve a informação de escala exigida pelo método |
| Tokenizador incompatível | IDs representam tokens diferentes | Distribua e valide o tokenizador correspondente |
| Arquivo escrito parcialmente | Erro de leitura ou dados truncados | Use escrita temporária e validação antes da publicação |
| Apenas tamanho em disco é comparado | Velocidade ou qualidade podem piorar sem serem percebidas | Avalie inferência e qualidade no runtime alvo |

## Resumo

- GGUF é um formato de inferência, mas suporte a arquitetura e quantização depende do runtime.
- Conversão exige mapear tensores e metadados; trocar a extensão não basta.
- Quantização ternária e empacotamento em dois bits são opções, não exigências de GGUF.
- Valide estrutura, tokenizador, comportamento, qualidade e desempenho após converter.
