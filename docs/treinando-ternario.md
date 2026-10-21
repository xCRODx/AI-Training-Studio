# Treinamento de redes com pesos ternários

Uma rede com pesos ternários restringe certos parâmetros a três valores, como $-1$, $0$ e $+1$. Durante o treino, porém, é comum manter pesos latentes contínuos e quantizá-los no cálculo do *forward pass*. Essa separação permite que o otimizador faça atualizações pequenas mesmo quando os pesos usados no cálculo são discretos.

## 1. A quantização e o problema do gradiente

Uma regra didática para quantizar uma matriz de pesos $W$ é:

$$
\alpha = \operatorname{mean}(|W|), \qquad
\widetilde{W} = \operatorname{clip}\!\left(\operatorname{round}(W/\alpha), -1, 1\right).
$$

O divisor $\alpha$ ajusta a escala antes do arredondamento. A forma exata de calcular a escala e os limites pode variar entre métodos. Se $\alpha=0$, a regra precisa tratar separadamente o caso de pesos nulos para evitar divisão por zero.

`round` e `clip` não têm derivadas úteis em quase todos os pontos. Aplicar a regra diretamente e propagar sua derivada pode bloquear o aprendizado dos pesos latentes.

## 2. Straight-Through Estimator

O *Straight-Through Estimator* (STE) usa a quantização no *forward pass*, mas aproxima a derivada durante o *backward pass*. Uma forma comum em PyTorch é:

```python
scaled = weight / scale
quantized = scaled.round().clamp(-1, 1)
weight_ste = (quantized - scaled).detach() + scaled
```

Numericamente, `weight_ste` tem o valor de `quantized`. Na retropropagação, o termo quantizado é tratado como constante e o gradiente passa pela aproximação contínua `scaled`.

O STE é uma aproximação enviesada, não a derivada verdadeira da quantização. Ele torna o treino possível em muitos cenários, mas não garante convergência ou qualidade. Existem outras aproximações e variantes de quantização.

## 3. Papel do otimizador

Otimizadores adaptativos como AdamW mantêm médias móveis do gradiente e de seu quadrado:

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t, \qquad
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2.
$$

Após as correções de viés, a atualização adaptativa é proporcional a:

$$
\frac{\widehat{m}_t}{\sqrt{\widehat{v}_t}+\epsilon}.
$$

Essa adaptação pode ajudar a lidar com gradientes ruidosos, mas não remove o viés introduzido pelo STE nem torna o passo de cada parâmetro constante. SGD com momentum e outros otimizadores também podem ser usados; a escolha deve ser avaliada empiricamente.

No AdamW, o *weight decay* é desacoplado da atualização adaptativa. Ele tende a reduzir a magnitude dos pesos latentes, mas a interação com a escala $\alpha$, os gradientes e a regra de quantização significa que a proporção de zeros não precisa aumentar monotonamente.

## 4. Inicialização e escala

Uma inicialização deve manter ativações e gradientes em faixas adequadas à arquitetura. Métodos como Xavier e He são pontos de partida comuns, mas nenhuma inicialização é universal para todas as redes quantizadas.

Como a escala de quantização depende dos pesos, monitore tanto os parâmetros latentes quanto a distribuição quantizada. Uma escala muito pequena também precisa de proteção numérica; uma escala excessiva pode alterar quantos pesos caem em cada estado.

## 5. Taxa de aprendizado e estabilidade

A taxa de aprendizado controla a velocidade com que os pesos latentes mudam. Uma taxa alta pode fazer muitos valores cruzarem os limiares de quantização de forma instável; uma taxa baixa pode tornar o aprendizado lento. Warmup e decaimento podem ser úteis, mas seus comprimentos e formatos devem corresponder à duração e ao objetivo do treino.

O parâmetro $\epsilon$ do otimizador evita divisões por zero e influencia a atualização quando os momentos são pequenos. Seu valor deve ser escolhido considerando a precisão numérica, a implementação do otimizador e a escala dos gradientes. Não existe um valor recomendado para todos os treinos em `bfloat16` ou `float16`.

## 6. Um ciclo de treino ilustrativo

```text
para cada lote:
    calcular previsões usando pesos quantizados e STE
    calcular a função de perda
    retropropagar o gradiente aproximado
    opcionalmente limitar a norma do gradiente
    atualizar os pesos latentes com o otimizador
    atualizar o scheduler, se houver
```

O código real depende da tarefa e pode incluir acumulação de gradientes, precisão mista, máscaras, distribuição entre dispositivos e tratamento de lotes incompletos.

## 7. O que monitorar

Conte os estados **depois da quantização**, não apenas os valores latentes. Para $N$ pesos quantizados:

$$
p_{-1}=\frac{\#\{i:\widetilde{w}_i=-1\}}{N},\quad
p_0=\frac{\#\{i:\widetilde{w}_i=0\}}{N},\quad
p_{+1}=\frac{\#\{i:\widetilde{w}_i=+1\}}{N}.
$$

As proporções somam $1$ (ou $100\%$ quando apresentadas como porcentagens). A proporção de zeros descreve esparsidade na representação quantizada, mas não implica aceleração automática: isso depende de kernels e formatos que consigam explorar zeros.

Interprete essas proporções junto da perda de treino e validação, norma dos gradientes, taxa de aprendizado e métricas da tarefa. Não há uma faixa de distribuição universalmente saudável. Muitos zeros podem reduzir capacidade; poucos zeros podem diminuir a compressão, mas o impacto depende do modelo e do objetivo.

## 8. Erros comuns

1. **Usar a derivada literal de `round`.** O gradiente pode desaparecer; um STE ou outra aproximação precisa ser escolhido conscientemente.
2. **Tratar o STE como gradiente exato.** Ele é uma aproximação e pode introduzir viés.
3. **Assumir que AdamW corrige qualquer instabilidade.** Taxa de aprendizado, dados, precisão e escala continuam importantes.
4. **Supor que mais zeros é sempre melhor.** Esparsidade pode diminuir a capacidade e só acelera a execução quando o runtime a aproveita.
5. **Copiar hiperparâmetros de outro modelo sem validar.** Valores adequados variam com a arquitetura, o conjunto de dados e o hardware.

## Resumo

- Pesos latentes contínuos podem ser quantizados no *forward pass* e atualizados com gradientes aproximados.
- O STE deixa o gradiente fluir, mas não é a derivada real da quantização.
- Otimizador, inicialização, precisão e taxa de aprendizado precisam ser avaliados em conjunto.
- A distribuição de pesos quantizados é uma medida descritiva, não uma métrica de qualidade isolada.
- Esparsidade só reduz custo de inferência quando a implementação e o hardware conseguem explorá-la.
