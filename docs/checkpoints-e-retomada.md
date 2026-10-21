# Checkpoints e retomada do treinamento

Treinamentos longos podem ser interrompidos por falhas, limites de tempo ou decisões de experimentação. Um checkpoint bem planejado permite continuar sem descartar o trabalho já feito. Para isso, salvar apenas os pesos pode não bastar: é preciso definir que tipo de retomada se deseja.

## 1. Três níveis de retomada

1. **Somente inferência:** pesos e configuração do modelo. Serve para executar o modelo, não para reproduzir o estado do treino.
2. **Continuação aproximada:** pesos, estado do otimizador e progresso. O treino continua, mas a sequência de exemplos ou de operações aleatórias pode mudar.
3. **Reprodução tão próxima quanto possível:** além dos itens anteriores, estado do scheduler, aleatoriedade, precisão mista, amostrador ou iterador de dados e demais estados relevantes.

Mesmo o terceiro nível não garante resultados bit a bit idênticos em qualquer máquina. Operações não determinísticas, versões de bibliotecas, hardware e ordem dos dados podem alterar os resultados.

## 2. O que pode ser salvo

- **Pesos e buffers do modelo:** o `state_dict` do modelo.
- **Otimizador:** momentos e contadores internos. No AdamW, isso inclui estimativas móveis dos gradientes e de seus quadrados.
- **Scheduler:** estado da taxa de aprendizado e posição no cronograma.
- **Precisão mista:** estado do `GradScaler`, quando usado.
- **Progresso:** passo do otimizador, época, tokens ou exemplos processados e configuração relevante.
- **Aleatoriedade:** estados dos geradores de Python, NumPy, PyTorch e, quando necessário, de cada dispositivo CUDA.
- **Ordem dos dados:** estado do sampler, do dataloader ou do iterador de streaming, se a biblioteca utilizada permitir serializá-lo.

A necessidade de cada item depende do objetivo. Para apenas carregar um modelo, normalmente bastam os pesos e a configuração da arquitetura. Para retomar o otimizador sem reinicializá-lo, salve também seu estado.

## 3. Por que o estado do otimizador importa

O AdamW mantém médias móveis dos gradientes e de seus quadrados:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2.
$$

Se apenas os pesos forem restaurados, o otimizador começa com $m_t$ e $v_t$ zerados. As atualizações seguintes podem diferir das que ocorreriam sem interrupção, mesmo que a taxa de aprendizado e os demais parâmetros sejam iguais.

O scheduler também tem estado próprio. Recriá-lo sem restaurar seu progresso pode reiniciar um warmup ou deslocar o decaimento da taxa de aprendizado.

## 4. Exemplo com PyTorch

O exemplo abaixo ilustra um checkpoint de continuação. Adapte-o ao otimizador, scheduler, dispositivos e fluxo de dados usados no treinamento:

```python
import os
import torch

state = {
    "step": step,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler is not None else None,
    "torch_rng": torch.get_rng_state(),
}

if torch.cuda.is_available():
    state["cuda_rng"] = torch.cuda.get_rng_state_all()

temporary_path = "checkpoint.tmp"
torch.save(state, temporary_path)
os.replace(temporary_path, "checkpoint.pt")
```

`os.replace` reduz a chance de deixar o arquivo final parcialmente escrito, desde que o temporário e o destino estejam no mesmo sistema de arquivos. Em configurações distribuídas ou com vários arquivos, use uma estratégia atômica adequada ao armazenamento e registre um manifesto que indique quando o conjunto está completo.

Para retomar, construa primeiro a mesma arquitetura e os objetos de treino, carregue os estados e restaure os geradores relevantes:

```python
state = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])

if scheduler is not None and state["scheduler"] is not None:
    scheduler.load_state_dict(state["scheduler"])

torch.set_rng_state(state["torch_rng"])
if "cuda_rng" in state and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(state["cuda_rng"])

step = state["step"]
```

Arquivos produzidos com `torch.save` podem conter dados serializados com pickle. Carregue-os apenas de fontes confiáveis. Para distribuir somente pesos, formatos como safetensors evitam a execução de objetos Python durante a leitura; eles não substituem, por si sós, o estado do otimizador e dos demais componentes.

## 5. Retomada do fluxo de dados

Salvar o passo do treino não informa necessariamente qual exemplo deve vir a seguir. Em um conjunto embaralhado, o fluxo depende da semente, do estado do sampler, dos trabalhadores do dataloader e de como as amostras são agrupadas.

Quando o sampler ou iterador oferece uma interface de serialização, salve e restaure seu estado. Se isso não for possível, registre as sementes, a configuração de embaralhamento e a posição aproximada. Reprocessar ou pular exemplos pode ser aceitável, mas não equivale a uma retomada exata — sobretudo em datasets em streaming.

## 6. Organização e integridade

Uma organização possível separa pesos para inferência e estado completo para treino:

```text
checkpoints/
├── latest/
│   ├── model.safetensors
│   ├── training_state.pt
│   └── metadata.json
└── step-050000/
    ├── model.safetensors
    ├── training_state.pt
    └── metadata.json
```

O nome e a estrutura dos arquivos são escolhas de organização, não um padrão obrigatório. Metadados úteis incluem arquitetura, dimensões, passo, tokenizador, configuração de treino, versões de bibliotecas e identificadores ou hashes dos dados.

Boas práticas:

- grave primeiro em arquivos temporários e publique o checkpoint completo somente após validar a escrita;
- mantenha cópias históricas se precisar comparar ou reverter estados;
- teste periodicamente a leitura e a retomada, não apenas o salvamento;
- documente quais estados foram preservados e quais não podem ser retomados exatamente;
- proteja os arquivos, pois checkpoints podem conter dados ou artefatos sensíveis.

## 7. Memória ocupada

Para $P$ parâmetros, um tensor de pesos em `float32` ocupa aproximadamente $4P$ bytes; em `bfloat16` ou `float16`, aproximadamente $2P$ bytes. O estado do AdamW pode acrescentar cerca de $8P$ bytes para dois momentos em `float32`, além de pesos, gradientes, buffers e outros estados. A ocupação real depende das opções do otimizador e da precisão usada.

Essa estimativa ajuda a planejar armazenamento e memória, mas não determina o tamanho exato do checkpoint comprimido ou do processo de treino.

## 8. Erros comuns

| Problema | Efeito possível | Prevenção |
|---|---|---|
| Salvar apenas os pesos e esperar continuar o mesmo treino | Estado do otimizador e scheduler reiniciados | Defina se o objetivo é inferência ou retomada e salve os estados necessários |
| Restaurar os pesos sem validar a arquitetura | Chaves ou dimensões incompatíveis | Salve metadados e valide a configuração antes de carregar |
| Ignorar sampler e aleatoriedade | Ordem diferente dos exemplos e resultados distintos | Salve os estados disponíveis e documente as limitações |
| Gravar diretamente sobre o único checkpoint | Falha pode corromper o arquivo anterior | Grave em temporário, valide e substitua de forma atômica |
| Carregar um arquivo serializado de origem desconhecida | Risco de executar conteúdo não confiável | Use fontes confiáveis e formatos de pesos apropriados |

## Resumo

- Pesos bastam para inferência, mas normalmente não para retomar o mesmo estado de otimização.
- Um checkpoint de treino pode incluir modelo, otimizador, scheduler, precisão mista, progresso, aleatoriedade e estado dos dados.
- A ordem do fluxo de dados é parte importante da reprodução de um experimento.
- Retomada bit a bit idêntica não é garantida entre ambientes diferentes.
- Integridade, metadados e testes de leitura são tão importantes quanto o ato de salvar.
