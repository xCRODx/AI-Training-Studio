# Distribuição de pesos ternários

Quando uma matriz de pesos é quantizada para $-1$, $0$ e $+1$, contar quantos valores caem em cada estado ajuda a descrever a representação resultante. Essa contagem é diferente da distribuição dos pesos latentes, que continuam sendo números contínuos durante o treinamento.

## 1. Definição

Para $N$ pesos quantizados $\widetilde{w}_i$, as proporções são:

$$
p_{-1}=\frac{\#\{i:\widetilde{w}_i=-1\}}{N},\qquad
p_0=\frac{\#\{i:\widetilde{w}_i=0\}}{N},\qquad
p_{+1}=\frac{\#\{i:\widetilde{w}_i=+1\}}{N}.
$$

Elas somam $1$, ou $100\%$ quando expressas em porcentagem. A proporção $p_0$ é a esparsidade dos pesos quantizados incluídos na contagem.

## 2. Exemplo de cálculo

Considere uma regra de quantização didática:

$$
\alpha=\operatorname{mean}(|W|),\qquad
\widetilde{W}=\operatorname{clip}(\operatorname{round}(W/\alpha),-1,1).
$$

Para os pesos $[0.003,-0.021,0.012,-0.0004,0.050,-0.045,0.001,-0.002]$, a escala é $\alpha=0.0168$. Aplicando arredondamento e clipping, obtém-se $[0,-1,+1,0,+1,-1,0,0]$.

- $-1$: 2 de 8 pesos, ou $25\%$;
- $0$: 4 de 8 pesos, ou $50\%$;
- $+1$: 2 de 8 pesos, ou $25\%$.

As regras de escala, agrupamento, arredondamento e tratamento de empates variam entre métodos. O cálculo precisa seguir a mesma regra usada pelo modelo ou pelo runtime para representar os pesos.

## 3. Relação com esparsidade e eficiência

Uma proporção alta de zeros significa que a matriz quantizada contém muitos valores nulos. Isso não implica automaticamente menos operações ou menor latência. Kernels densos podem continuar percorrendo todos os elementos; para acelerar, o runtime precisa usar formatos ou instruções que explorem os zeros.

Esparsidade estruturada — por exemplo, blocos ou linhas inteiras zerados — costuma ser mais fácil de explorar do que zeros espalhados individualmente, mas o benefício depende do suporte de hardware e software.

## 4. Interpretação durante o treino

Uma evolução na proporção de zeros pode resultar de mudanças nos pesos latentes, da escala de quantização ou da regra usada. Não há uma trajetória obrigatória nem uma faixa universal de valores saudáveis.

Observe as três proporções junto com a perda de validação, as métricas da tarefa e as normas de pesos e gradientes. Uma distribuição equilibrada não garante boa qualidade; muitos zeros não significam necessariamente perda de capacidade; e uma assimetria entre $-1$ e $+1$ não é, por si só, prova de erro.

## 5. Implementação de uma contagem

O trecho abaixo mostra a lógica após quantizar uma matriz com a regra escolhida:

```python
import torch

def ternary_distribution(weight, scale):
    quantized = torch.round(weight / scale).clamp(-1, 1)
    total = quantized.numel()
    if total == 0:
        return {"-1": 0.0, "0": 0.0, "+1": 0.0}

    return {
        str(value): (quantized == value).sum().item() / total * 100
        for value in (-1, 0, 1)
    }
```

Em um modelo completo, selecione explicitamente quais tensores entram na medição. Inclua camadas comparáveis e documente exclusões, como biases, embeddings ou parâmetros mantidos em ponto flutuante.

## 6. Erros comuns

1. Contar pesos latentes em vez dos valores após quantização.
2. Supor que a fração de zeros se converte diretamente em aceleração.
3. Tratar uma distribuição visualmente equilibrada como prova de qualidade.
4. Comparar contagens que usam escalas, camadas ou regras de quantização diferentes.
5. Ignorar a perda de validação e as métricas da tarefa ao interpretar esparsidade.

## Resumo

- A distribuição ternária conta pesos quantizados em $-1$, $0$ e $+1$.
- A fração de zeros mede esparsidade da representação, não necessariamente economia de computação.
- Resultados dependem das regras de quantização e dos tensores incluídos.
- Interprete a distribuição junto das métricas de qualidade e do comportamento do runtime.
