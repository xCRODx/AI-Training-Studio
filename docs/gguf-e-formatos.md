# Formatos de modelos: checkpoints, safetensors e GGUF

Um modelo treinado precisa ser armazenado em um formato que preserve os dados necessários para seu próximo uso. O formato mais apropriado depende de se o objetivo é continuar o treinamento, distribuir pesos com segurança ou executar inferência em um runtime específico.

## 1. Checkpoints de treinamento

Um checkpoint pode conter pesos, buffers do modelo, estado do otimizador, progresso, scheduler, estado aleatório e configuração. A estrutura varia entre bibliotecas e projetos.

Em PyTorch, `state_dict` é um mapeamento de nomes para tensores. Um arquivo de checkpoint pode incluir mais componentes:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
}
torch.save(checkpoint, "checkpoint.pt")
```

`torch.save` usa serialização Python. Arquivos de origem desconhecida devem ser tratados com cautela: carregá-los com opções de desserialização que aceitam objetos arbitrários pode executar código. Quando possível, use as opções seguras da biblioteca para pesos ou formatos voltados apenas a tensores.

## 2. Safetensors

Safetensors é um formato para armazenar tensores e seus metadados sem serializar objetos Python arbitrários. Ele pode ser útil para distribuir pesos e carregá-los em ferramentas compatíveis.

O formato não inclui, por si só, o estado completo do otimizador ou do fluxo de dados. Portanto, pode servir para inferência ou transferência de pesos, mas não substitui todos os elementos de um checkpoint de treinamento.

## 3. GGUF

GGUF é um formato binário utilizado por diversos runtimes de inferência, incluindo ferramentas do ecossistema llama.cpp. Ele pode armazenar metadados, tensores e tipos numéricos ou quantizados.

Um arquivo GGUF não é automaticamente executável em qualquer programa. O runtime precisa reconhecer a arquitetura, os metadados, os nomes e dimensões dos tensores, os tipos utilizados e o tokenizador esperado. Dependendo do modelo e do arquivo, alguns componentes também podem ser fornecidos separadamente.

GGUF é voltado principalmente à inferência, não a preservar todo o estado necessário para continuar o treinamento.

## 4. Comparação de objetivos

| Aspecto | Checkpoint de treino | Safetensors | GGUF |
|---|---|---|---|
| Uso comum | Retomar ou analisar o treino | Armazenar e distribuir tensores | Inferência em runtimes compatíveis |
| Estado do otimizador | Pode incluir | Não é seu objetivo | Geralmente não |
| Serialização de objetos Python | Pode ocorrer, conforme o conteúdo | Não | Não |
| Tokenizador e configuração | Dependem do checkpoint | Normalmente externos | Podem estar em metadados ou ser externos |
| Compatibilidade | Depende da biblioteca e arquitetura | Depende da ferramenta que lê os tensores | Depende do suporte do runtime |
| Quantização | Possível, conforme a ferramenta | Não define um método de quantização | Pode armazenar tensores quantizados compatíveis |

## 5. Precisão e tamanho

Para $N$ valores, o payload bruto ocupa aproximadamente $4N$ bytes em `float32`, $2N$ em `float16` ou `bfloat16`, e $N/4$ bytes se cada valor puder ser armazenado em dois bits. Esses cálculos ignoram metadados, alinhamento, escalas e tensores mantidos em maior precisão.

Tipos quantizados, escalas e layouts são específicos do formato e do método. A redução do arquivo não garante a mesma redução de memória durante a inferência nem uma aceleração proporcional; isso depende do runtime e do hardware.

## 6. Escolha do formato

- Para **retomar treinamento**, preserve os estados de treino necessários e use formatos confiáveis.
- Para **distribuir pesos**, prefira formatos focados em tensores e acompanhe-os de configuração e tokenizador.
- Para **inferência**, escolha um formato aceito pelo runtime pretendido e valide o modelo convertido.

## Erros comuns

- Tratar formatos diferentes como se fossem intercambiáveis.
- Presumir que todos os metadados e o tokenizador estão embutidos.
- Carregar checkpoints de origem desconhecida sem considerar os riscos da serialização.
- Supor que qualquer runtime interpreta qualquer variante de GGUF.
- Comparar tamanhos sem validar qualidade e desempenho.

## Resumo

- Checkpoints de treinamento podem preservar estado além dos pesos.
- Safetensors armazena tensores sem serializar objetos Python arbitrários.
- GGUF organiza pesos e metadados para runtimes de inferência compatíveis.
- Compatibilidade, segurança, quantização e tokenizador precisam ser verificados para cada uso.
