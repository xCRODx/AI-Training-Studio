
# Configuração reproduzível de experimentos

Experimentos de machine learning dependem de muitos valores: tamanho de lote, taxa de aprendizado, caminho dos dados, dispositivo de execução e critérios de validação. Separar essas configurações do código facilita comparar execuções e reduz alterações acidentais.

## Parâmetros, configuração e segredos

Uma configuração reúne valores que controlam uma execução. Agrupe-os por finalidade e registre os valores usados em cada experimento:

```yaml
model:
  hidden_size: 512
  layers: 8
  context_length: 1024
training:
  learning_rate: 0.0003
  batch_size: 8
  seed: 42
data:
  train_path: data/train.jsonl
  validation_path: data/validation.jsonl
```

Os valores acima são ilustrativos, não recomendações universais. Uma configuração adequada depende da arquitetura, dos dados, do objetivo e do hardware.

## Variáveis de ambiente

Variáveis de ambiente são pares `NOME=valor` fornecidos ao processo pelo sistema operacional. São úteis para alternar caminhos, dispositivos e credenciais sem editar o código:

```powershell
$env:DEVICE = "cuda"
$env:LEARNING_RATE = "0.0003"
python train.py
```

```bash
DEVICE=cuda LEARNING_RATE=0.0003 python train.py
```

Em Python, os valores podem ser lidos com `os.environ`. Como variáveis de ambiente são texto, converta e valide cada valor antes de usá-lo:

```python
import os

device = os.getenv("DEVICE", "cpu")
learning_rate = float(os.getenv("LEARNING_RATE", "0.0003"))
if learning_rate <= 0:
    raise ValueError("LEARNING_RATE deve ser maior que zero")
```

Arquivos `.env` são uma convenção prática para carregar pares de variáveis localmente, geralmente por meio de uma biblioteca. O Python não os lê automaticamente. Não inclua arquivos com segredos em controle de versão nem os envie para serviços que não devam recebê-los.

## Como manter experimentos comparáveis

1. **Defina valores padrão explícitos.** Documente quais valores são usados quando não há configuração fornecida.
2. **Valide entradas.** Verifique tipos, intervalos e combinações incompatíveis antes de iniciar um treinamento longo.
3. **Registre a execução.** Salve configurações junto dos resultados, métricas, versão dos dados e identificador do checkpoint.
4. **Controle as sementes.** Uma semente ajuda a repetir partes aleatórias, mas não garante resultados idênticos em todas as bibliotecas e dispositivos.
5. **Separe configuração de credenciais.** Tokens e chaves são segredos; taxas de aprendizado e tamanhos de lote não são.
6. **Evite valores implícitos.** Se uma opção altera arquitetura, precisão ou divisão dos dados, registre-a junto dos resultados.

## Checklist antes do treinamento

- Os caminhos dos conjuntos de treino e validação apontam para os dados esperados?
- O tamanho do lote e o comprimento de contexto cabem na memória disponível?
- A taxa de aprendizado, a precisão numérica e o otimizador são compatíveis?
- As credenciais estão fora dos arquivos públicos e dos logs?
- A configuração completa será salva com o checkpoint?

Uma execução reproduzível não é apenas um conjunto de números: inclui também os dados, o código, as bibliotecas, as sementes e as condições de hardware relevantes.
