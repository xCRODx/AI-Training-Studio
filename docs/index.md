# Inteligência artificial: fundamentos e prática

Uma trilha de aprendizado progressiva sobre inteligência artificial, machine learning, redes neurais e modelos de linguagem. Os tópicos combinam explicações intuitivas, exemplos numéricos, matemática, código e diagramas.

## Caminho de aprendizagem

Para começar do zero, siga esta sequência:

1. [O que é aprendizado de máquina](./o-que-e-machine-learning.md)
2. [Probabilidade para quem nunca viu](./probabilidade.md)
3. [Vetores, matrizes e tensores](./vetores-e-produto-escalar.md)
4. [O neurônio artificial](./neuronio-artificial.md)
5. [Funções de ativação](./funcoes-de-ativacao.md)
6. [Função de perda](./funcao-de-perda.md)
7. [Gradientes e backpropagation](./gradientes.md)
8. [Descida do gradiente e AdamW](./descida-do-gradiente.md)
9. [O que é um modelo de linguagem](./o-que-e-um-modelo-de-linguagem.md)
10. [Transformer, RetNet e outras arquiteturas](./comparativo-arquiteturas.md)

## 1. Inteligência artificial e machine learning

- [Aprendizado de máquina: conceitos essenciais](./o-que-e-machine-learning.md)
- [Probabilidade para quem nunca viu](./probabilidade.md)
- [Estatística: média, variância e desvio padrão](./estatistica-basica.md)
- [Overfitting, underfitting e regularização](./overfitting-e-regularizacao.md)

## 2. Matemática para redes neurais

- [Vetores e produto escalar](./vetores-e-produto-escalar.md)
- [Multiplicação de matrizes](./matrizes.md)
- [Tensores: formas, dimensões e operações](./tensores.md)
- [Álgebra linear essencial](./algebra-linear.md)
- [Derivadas e regra da cadeia](./derivadas.md)
- [Gradientes e derivadas parciais](./gradientes.md)
- [Logaritmos e exponenciais](./logaritmos-e-exponenciais.md)
- [Números no computador e precisão numérica](./numeros-no-computador.md)

## 3. Como redes neurais aprendem

- [O neurônio artificial](./neuronio-artificial.md)
- [Camadas lineares e embeddings](./camadas-lineares-e-embeddings.md)
- [Funções de ativação](./funcoes-de-ativacao.md)
- [Normalização: LayerNorm e RMSNorm](./normalizacao.md)
- [Forward pass](./forward-pass.md)
- [Função de perda](./funcao-de-perda.md)
- [Backpropagation](./backpropagation.md)
- [Autograd e grafos computacionais](./autograd.md)
- [Descida do gradiente](./descida-do-gradiente.md)
- [SGD e momentum](./sgd-e-momentum.md)
- [Adam e AdamW](./adam-e-adamw.md)
- [Learning rate e schedulers](./learning-rate-e-schedulers.md)
- [Gradient clipping](./gradient-clipping.md)
- [Acumulação de gradientes](./gradient-accumulation.md)
- [Precisão mista e bfloat16](./mixed-precision.md)
- [Dropout e outras regularizações](./dropout.md)
- [Métricas de treinamento](./metricas-de-treino.md)

## 4. Dados para inteligência artificial

- [Pré-processamento e tokenização](./preprocessamento-tokenizacao.md)
- [Tokenizers e vocabulários](./tokenizer.md)
- [Datasets em streaming](./dataset-streaming.md)
- [Datasets para fine-tuning](./dataset-finetuning.md)
- [Hugging Face Datasets e Transformers](./huggingface.md)

## 5. Modelos de linguagem

- [O que é um modelo de linguagem](./o-que-e-um-modelo-de-linguagem.md)
- [Geração autorregressiva](./geracao-autorregressiva.md)
- [Sampling: temperatura, top-p e top-k](./sampling.md)
- [Logits, softmax e entropia cruzada](./softmax-e-entropia-cruzada.md)
- [Atenção: queries, keys e values](./atencao-qkv.md)
- [Multi-head attention](./multi-head-attention.md)
- [Codificação posicional e RoPE](./codificacao-posicional.md)
- [KV cache e estado recorrente](./kv-cache.md)
- [Como modelos representam e usam informação](./como-o-modelo-guarda.md)

## 6. Arquiteturas de sequência

- [O que é o Transformer](./o-que-e-transformer.md)
- [O que é o RetNet](./o-que-e-retnet.md)
- [Matemática da retenção](./retention-matematica.md)
- [RNN, LSTM e GRU](./rnn-lstm-gru.md)
- [RWKV](./rwkv.md)
- [Mamba e modelos de espaço de estado](./mamba-ssm.md)
- [Comparação entre arquiteturas](./comparativo-arquiteturas.md)

## 7. Quantização e eficiência

- [O que é o BitNet](./o-que-e-bitnet.md)
- [Representação numérica e quantização](./representacao-quantizada.md)
- [BitLinear em detalhe](./bitlinear.md)
- [Straight-Through Estimator](./ste.md)
- [Quantização de ativações](./quantizacao-ativacoes.md)
- [Escalas alfa e beta](./escalas-alpha-beta.md)
- [BitNet e pesos em ponto flutuante](./bitnet-vs-fp.md)
- [Distribuição ternária dos pesos](./distribuicao-ternaria.md)
- [Treinamento de modelos ternários](./treinando-ternario.md)
- [Quantização com bitsandbytes](./bitsandbytes.md)

## 8. Treinar e adaptar modelos

- [Pré-treinamento e fine-tuning](./pre-treino-vs-finetuning.md)
- [Pipeline de treinamento](./pipeline-treinamento.md)
- [O que é fine-tuning](./o-que-e-finetuning.md)
- [Supervised fine-tuning (SFT)](./sft.md)
- [LoRA e PEFT](./lora-e-peft.md)
- [Configuração reproduzível de experimentos](./env-variaveis.md)
- [Contexto, batch size e acumulação](./contexto-batch-acumulacao.md)
- [Acumulação de gradientes](./gradient-accumulation.md)
- [Gradient checkpointing e memória de ativações](./gradient-checkpointing.md)
- [Weight decay, warmup e norma do gradiente](./weight-decay-warmup.md)
- [Checkpoints e retomada do treinamento](./checkpoints-e-retomada.md)
- [Throughput e planejamento de treinamento](./throughput-e-marcos.md)

## 9. Inferência e execução

- [Inferência em CPU e GPU](./inferencia-cpu-gpu.md)
- [Precisão numérica na inferência](./precisao-na-inferencia.md)
- [CUDA e GPUs](./cuda-e-gpus.md)
- [PyTorch para modelos de linguagem](./pytorch-llm.md)
- [Formatos de pesos e checkpoints](./gguf-e-formatos.md)
- [Implantação de modelos](./runtime-deploy.md)
- [safetensors](./safetensors.md)
