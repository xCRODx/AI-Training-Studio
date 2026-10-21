# Throughput, tokens por segundo e marcos de treino

Para planejar um treinamento, é útil acompanhar tanto a quantidade de trabalho já feita quanto a velocidade observada. Essas medidas ajudam a estimar duração e comparar execuções, mas não predizem, sozinhas, a qualidade do modelo.

## 1. O que medir

- **Tokens processados:** total acumulado segundo uma convenção definida. Registre se a contagem inclui padding, tokens de entrada ou alvos.
- **Tokens por segundo:** tokens contados divididos pelo tempo decorrido na mesma janela de medição.
- **Lotes por segundo:** lotes processados por segundo. Especifique se “lote” significa microbatch ou atualização do otimizador.
- **Tempo de parede:** inclui leitura de dados, validação, checkpoints, pausas e outros custos da execução.

Para lotes de sequências de comprimento uniforme, um microbatch com $B$ sequências de $L$ posições contém $B L$ posições de token. Se houver padding, descarte de posições ou múltiplos dispositivos, a contagem de tokens úteis pode ser diferente. Na acumulação de gradientes, vários microbatches formam uma atualização, mas cada um consome tempo e processa tokens.

Uma decomposição simples é:

$$
\text{tokens/s} = \text{microbatches/s} \times \text{tokens contados por microbatch}
$$

Ela só vale quando os dois termos usam a mesma janela e convenção de contagem. Uma queda de throughput não identifica, por si só, a causa: pode vir do modelo, do hardware, da entrada de dados, de comunicação ou de outra sobrecarga.

## 2. Estimar o tempo restante

Se faltam $T_r$ tokens e a taxa média observada é $v$ tokens/s, uma estimativa simples é:

$$
\text{ETA} = \frac{T_r}{v}
$$

Em geral, é melhor calcular $v$ numa janela recente ou suavizar medições sucessivas. A estimativa deve ser atualizada quando o throughput muda e interpretada como aproximação, não como horário garantido. Validações e gravações de checkpoint acrescentam tempo que uma taxa medida apenas durante os passos de treino pode não incluir.

Uma média móvel exponencial pode reduzir oscilações de curto prazo:

$$
\bar{v}_t = \alpha v_t + (1-\alpha)\bar{v}_{t-1}, \qquad 0 < \alpha \le 1
$$

Um valor menor de $\alpha$ suaviza mais, mas reage mais lentamente a mudanças. A janela de medição e o valor de $\alpha$ devem ser escolhidos de acordo com a frequência de atualização.

```python
import time

class ThroughputTracker:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.last_tokens = None
        self.last_time = time.perf_counter()
        self.rate = None

    def update(self, tokens_processed):
        now = time.perf_counter()
        if self.last_tokens is not None:
            elapsed = now - self.last_time
            delta = tokens_processed - self.last_tokens
            if elapsed > 0 and delta >= 0:
                instant_rate = delta / elapsed
                if self.rate is None:
                    self.rate = instant_rate
                else:
                    self.rate = (
                        self.alpha * instant_rate
                        + (1 - self.alpha) * self.rate
                    )

        self.last_tokens = tokens_processed
        self.last_time = now
        return self.rate

    def eta_seconds(self, target_tokens, tokens_processed):
        if self.rate is None or self.rate <= 0:
            return None
        remaining = max(0, target_tokens - tokens_processed)
        return remaining / self.rate
```

O contador passado a `update` deve ser acumulado de forma consistente e não voltar para zero entre chamadas. Em execuções distribuídas, conte tokens de todos os dispositivos uma única vez, não uma vez por processo.

## 3. Marcos de avaliação

Um marco é um ponto planejado para inspecionar o andamento: por exemplo, após uma quantidade de tokens, um número de atualizações ou um intervalo de tempo. Em cada marco, pode ser útil:

1. medir a perda em dados de validação;
2. gerar amostras com um procedimento fixo;
3. salvar estado suficiente para retomar ou comparar a execução;
4. decidir se continua, ajusta a configuração ou encerra.

Não há uma lista universal de quantidades de tokens que garanta determinado nível de qualidade. O orçamento, a arquitetura, os dados e o objetivo variam. Defina marcos adequados à pergunta do experimento e compare-os usando os mesmos dados e métricas.

## 4. Exemplo de cálculo

Suponha que uma execução conte 12 milhões de tokens em 10 minutos:

$$
v = \frac{12\,000\,000}{10 \times 60} = 20\,000\;\text{tokens/s}
$$

Se faltarem 300 milhões de tokens, a estimativa baseada nessa taxa é:

$$
\text{ETA} = \frac{300\,000\,000}{20\,000} = 15\,000\;\text{s} \approx 4.17\;\text{h}
$$

Essa projeção pressupõe que a taxa se mantenha e não inclui necessariamente validação, escrita de checkpoints, preparação de dados ou interrupções. Uma medição mais longa e representativa costuma produzir uma estimativa melhor.

## 5. Comparar configurações com cuidado

Ao comparar duas execuções, mantenha constantes, tanto quanto possível, o hardware, os dados, o comprimento das sequências, o número de tokens úteis, a precisão e o método de medição. Registre também os parâmetros que mudaram. Uma configuração que processa mais tokens por segundo não é automaticamente melhor se alterar a qualidade, a estabilidade ou o custo total.

Para diagnosticar gargalos, observe separadamente:

- tempo de preparação e espera por dados;
- tempo de cálculo do modelo;
- tempo de comunicação entre dispositivos, se houver;
- pausas para avaliação e gravação;
- memória, temperatura e utilização do hardware.

Faça medições repetidas após o aquecimento do runtime, mas não descarte o custo de inicialização se ele fizer parte do uso real que deseja estimar.

## Erros comuns

- **Usar a velocidade do primeiro intervalo como previsão definitiva.** O ritmo pode mudar; atualize a medição.
- **Confundir microbatch com atualização do otimizador.** Acumulação de gradientes altera a relação entre lotes, passos e tokens.
- **Misturar convenções de contagem.** Padding, tokens de entrada e alvos podem ser contados de maneiras diferentes.
- **Ignorar o tempo de parede.** Uma taxa medida apenas durante o cálculo não inclui todas as pausas do treinamento.
- **Tratar marcos como previsões de qualidade.** Eles são pontos de inspeção; avalie o modelo nos dados e tarefas relevantes.

## Resumo

Throughput descreve trabalho por tempo; ETA projeta o tempo restante a partir de uma taxa medida. Defina a convenção de contagem, use medições representativas, inclua os custos de parede e trate os marcos como oportunidades de avaliação — não como garantias de qualidade.
