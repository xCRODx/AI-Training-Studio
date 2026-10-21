# Pesos ternários e pesos em ponto flutuante

Uma camada linear calcula produtos entre entradas e pesos. Com pesos em ponto flutuante, cada termo pode ter uma magnitude arbitrária dentro da precisão disponível. Com pesos ternários, cada termo usa um dos valores $-1$, $0$ e $+1$, geralmente com escalas adicionais para aproximar magnitudes.

## 1. Produto denso

Para uma entrada $x$ e pesos $w$, uma saída é:

$$
y=\sum_{i=1}^{n}x_iw_i.
$$

Com pesos em ponto flutuante, cada termo usa uma multiplicação seguida de acumulação. Bibliotecas de álgebra linear executam essas operações com kernels especializados; o número de instruções e o consumo variam entre hardware e formatos.

## 2. Produto com pesos ternários

Se $q_i\in\{-1,0,+1\}$, então:

$$
x_iq_i=\begin{cases}
x_i,&q_i=+1,\\
0,&q_i=0,\\
-x_i,&q_i=-1.
\end{cases}
$$

Assim, o produto pode ser descrito como uma soma de entradas com sinal. Uma variante pode incluir uma escala $\alpha$:

$$
y\approx\alpha\sum_i x_iq_i.
$$

Essa forma matemática não garante que um processador elimine multiplicações no código de máquina. A representação, o tipo das ativações, o kernel e o hardware determinam quais instruções são executadas.

## 3. Exemplo numérico

Para $x=[2,3,-1,4]$ e pesos em ponto flutuante $w=[0.7,-0.5,0.3,0.9]$:

$$
x\cdot w=2(0.7)+3(-0.5)+(-1)(0.3)+4(0.9)=3.2.
$$

Com pesos ternários $q=[+1,-1,0,+1]$ e sem escala:

$$
x\cdot q=2-3+0+4=3.
$$

O exemplo mostra que a saída muda quando os pesos são quantizados. Uma escala pode ajustar a magnitude, mas não elimina todo erro de quantização.

## 4. Memória e codificação

Um peso ternário tem três estados possíveis. A quantidade de informação para representar estados equiprováveis é $\log_2 3\approx1.585$ bits por símbolo. Em armazenamento de comprimento fixo, dois bits por peso são uma codificação simples: quatro pesos cabem em um byte, com uma combinação não utilizada.

Uma camada com $P$ pesos em `float32` ocupa aproximadamente $4P$ bytes; com dois bits por peso, o payload ideal ocupa $P/4$ bytes, sem contar escalas, alinhamento, outros tensores ou metadados. A proporção de redução do modelo completo será menor quando houver pesos não quantizados.

## 5. O que se pode concluir sobre eficiência

Pesos ternários podem reduzir o armazenamento dos pesos e permitir kernels que aproveitam operações simples ou dados compactos. Eles não garantem uma aceleração ou economia de energia específica. Uma implementação pode precisar decodificar códigos, aplicar escalas ou continuar usando operações de ponto flutuante nas ativações.

O efeito final depende de:

- formato de empacotamento e escala;
- proporção de tensores quantizados;
- kernels e instruções disponíveis;
- largura de banda e memória do dispositivo;
- tamanho do lote e forma da carga de trabalho;
- qualidade alcançada após quantizar.

## 6. Comparação prática

Para comparar pesos ternários com `float16`, `bfloat16` ou `float32`, mantenha constante a arquitetura e avalie:

1. tamanho dos pesos e memória total de inferência;
2. latência do primeiro token, latência por token e throughput;
3. qualidade em validação e exemplos representativos;
4. estabilidade numérica e compatibilidade com o runtime.

Use medições do dispositivo de destino. Números de energia por operação publicados para uma geração de processo de fabricação não determinam o consumo de outro chip ou de um modelo completo.

## Erros comuns

- Afirmar que toda multiplicação desaparece quando os pesos são ternários.
- Confundir o limite teórico de entropia com o número prático de bits de armazenamento.
- Comparar apenas o tamanho dos pesos e ignorar escalas e outros tensores.
- Tratar zeros como operações economizadas sem verificar suporte a esparsidade.
- Inferir velocidade ou energia do formato sem medir o runtime.

## Resumo

- Pesos ternários permitem escrever produtos como somas com sinal e termos nulos.
- Dois bits por peso são uma codificação simples, mas não representam o tamanho total do modelo.
- Quantização altera os valores e pode afetar a qualidade.
- Benefícios de velocidade, memória e energia dependem da implementação e devem ser medidos.
