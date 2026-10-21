# AI Training Studio

Ambiente local para treinar, ajustar, avaliar e executar modelos de inteligência artificial, com suporte a pré-treinamento, fine-tuning supervisionado (SFT), quantização e exportação GGUF.

---

## 📋 Sumário
1. [Requisitos de Sistema](#-requisitos-de-sistema)
2. [Instalação Local (Bare-Metal / Venv)](#-instalação-local-bare-metal--venv)
3. [Validação do Ambiente](#-validação-do-ambiente)
4. [Configuração do Ambiente (.env.local)](#-configuração-do-ambiente-envlocal)
5. [Como Executar Localmente](#-como-executar-localmente)
6. [Execução com Docker e Docker Compose (Windows/Linux)](#-execução-com-docker-e-docker-compose-windowslinux)
7. [Mapeamento das Interfaces Web](#-mapeamento-das-interfaces-web)
8. [Estrutura do Projeto](#-estrutura-do-projeto)
9. [Solução de Problemas (Troubleshooting)](#-solução-de-problemas-troubleshooting)

---

## 💻 Requisitos de Sistema

### Hardware e memória
- **GPU**: uma GPU NVIDIA compatível com CUDA acelera o treinamento, mas não há um requisito mínimo universal de VRAM. A memória necessária varia conforme o tamanho do modelo, o contexto, o lote, a precisão numérica e o método de treinamento.
- **GPUs com menos memória**: também podem treinar modelos menores ou configurações mais econômicas. Reduzir o lote por microbatch e o contexto ajuda a diminuir o uso de memória; o treinamento pode levar mais tempo.
- **Equipamento usado nos testes**: os testes de treinamento foram realizados em uma NVIDIA GeForce RTX 5060 Ti com 16 GB de VRAM. Esse é apenas o hardware usado como referência de teste, não um requisito mínimo.
- **CPU**: há suporte para execução em CPU, útil para testes e inferência leve; o desempenho costuma ser menor que em uma GPU.
- **Memória RAM**: 16 GB ou superior é uma referência recomendada; a necessidade varia conforme os dados, o modelo e os processos executados ao mesmo tempo.
- **Armazenamento**: 20 GB de espaço livre em disco (SSD recomendado para datasets e checkpoints).

### Software Necessário
- **Sistema Operacional**: Linux (Ubuntu 20.04/22.04/24.04, Debian 11/12), WSL2 no Windows ou Windows 10/11 nativo.
- **Python**: Versão **3.10** ou **3.11** (recomendado 3.11).
- **Driver NVIDIA**: versão compatível com CUDA 12.8. O toolkit CUDA não precisa ser instalado no host quando o Docker é usado.
- **Ferramentas de build C++** (necessárias para os runtimes nativos):
  - Linux: `build-essential`, `cmake`, `ninja-build`.
  - Windows: Visual Studio Build Tools (C++ CMake).
- **Docker & Docker Compose** (para execução conteinerizada no Linux).

---

## 🛠 Instalação Local (Bare-Metal / Venv)

### 1. Clonar o Repositório
```bash
git clone <URL_DO_REPOSITORIO>
cd AI-Training-Studio
```

### 2. Criar e Ativar Ambiente Virtual

**No Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**No Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Atualizar o Pip e Instalar Dependências

Instale primeiro o PyTorch compilado com suporte a CUDA 12.4 e em seguida as dependências do projeto:

```bash
# Atualiza pip e ferramentas de build
python -m pip install --upgrade pip setuptools wheel

# Instala PyTorch oficial com suporte CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Instala os demais pacotes do projeto
pip install -r requirements.txt
```

---

## ✅ Validação do Ambiente

Para garantir que o PyTorch reconhece sua GPU e que todas as dependências estão corretas, execute o teste rápido no terminal:

```bash
python -c "import torch, transformers, fastapi, bitsandbytes; print('CUDA disponível:', torch.cuda.is_available()); print('Dispositivo:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Versão PyTorch:', torch.__version__)"
```

Saída esperada (com GPU):
```text
CUDA disponível: True
Dispositivo: NVIDIA GeForce RTX ...
Versão PyTorch: 2.4.x+cu124 (ou superior)
```

---

## ⚙ Configuração do Ambiente (.env.local)

O sistema utiliza a biblioteca `pydantic-settings` para carregar configurações dos arquivos `.env.local` e `.env`.

### Seleção de Arquitetura

Duas flags booleanas definem a arquitetura usada em **pré-treino, fine-tuning, inferência e compilação** (o pós-treino sempre herda a arquitetura gravada no `config.json` do checkpoint, garantindo a mesma arquitetura do pré-treino):

```env
# BitNet e RetNet são opções independentes. É possível ativar uma, as duas ou nenhuma.
# USE_BITNET=true  + USE_RETNET=true  -> BitNet + RetNet (pesos ternários e retenção)
# USE_BITNET=true  + USE_RETNET=false -> BitNet (pesos ternários e atenção Transformer)
# USE_BITNET=false + USE_RETNET=true  -> RetNet (retenção e projeções convencionais)
# USE_BITNET=false + USE_RETNET=false -> Transformer (atenção e projeções convencionais)
USE_BITNET=false
USE_RETNET=false

# GQA (opcional, apenas transformer/bitnet): nº de KV heads. 0 = automático.
MODEL_KV_HEADS=0
# RoPE (apenas transformer/bitnet): frequência base.
ROPE_THETA=10000.0
```

- **Transformer** → atenção convencional, com opções como RoPE, GQA, RMSNorm e KV cache.
- **BitNet** → quantização ternária dos pesos das camadas lineares; pode usar atenção Transformer.
- **RetNet** → mecanismo de retenção para processar sequências.
- **BitNet + RetNet** → combinação opcional de pesos ternários e retenção.

A compilação seleciona um runtime compatível com a arquitetura e o formato exportado. O runtime C++ nativo atende à combinação de pesos ternários e retenção; BitNet e Transformer também podem ser exportados para runtimes compatíveis.

Crie um arquivo `.env.local` na raiz do projeto (ou edite o existente):

Os valores abaixo são um exemplo, não uma configuração universal. Ajuste o tamanho do modelo, o contexto e os lotes à memória disponível; GPUs diferentes podem treinar modelos e configurações diferentes.

```env
# =======================================================
# Arquitetura do Modelo (~330M Parâmetros)
# =======================================================
MODEL_D_MODEL=1024
MODEL_LAYERS=24
MODEL_EXPERTS=4
CONTEXT_WINDOW=512

# =======================================================
# Desempenho e Treinamento
# =======================================================
TRAIN_BATCH_SIZE=6
GRADIENT_ACCUMULATION_STEPS=8
GRADIENT_CHECKPOINTING=true
TRAIN_LR=2.0e-4
TRAIN_EPOCHS=1
CHECKPOINT_SAVE_STEPS=1000
MAX_GRAD_NORM=1.0

# =======================================================
# Tokenizer e Dataset de Pré-treino
# =======================================================
INFERENCE_DEVICE=cuda
TOKENIZER_NAME=neuralmind/bert-base-portuguese-cased
DATASET_NAME=eduagarcia/CrawlPT_dedup
HF_TOKEN=seu_token_huggingface_aqui

# =======================================================
# Fine-Tuning Conversacional (SFT)
# =======================================================
FINE_TUNING_ENGINE=local
FINE_TUNING_METHOD=full # full ou lora; também selecionável no painel
FINE_TUNING_LORA_RANK=16
FINE_TUNING_LORA_ALPHA=32
FINE_TUNING_DATASET_NAME=HuggingFaceTB/everyday-conversations-llama3.1-2k
FINE_TUNING_MICRO_BATCH_SIZE=2
FINE_TUNING_GLOBAL_BATCH_SIZE=32
FINE_TUNING_GRADIENT_ACCUMULATION_STEPS=16
FINE_TUNING_LEARNING_RATE=2e-5
FINE_TUNING_EPOCHS=3
```

Para aliviar a memória, diminua `TRAIN_BATCH_SIZE` ou `FINE_TUNING_MICRO_BATCH_SIZE` e, se quiser manter um lote efetivo semelhante, aumente `GRADIENT_ACCUMULATION_STEPS` ou `FINE_TUNING_GRADIENT_ACCUMULATION_STEPS`. A acumulação processa micro-lotes menores antes de atualizar os pesos; sozinha, sem reduzir o micro-lote, ela não diminui o pico de memória. `GRADIENT_CHECKPOINTING=true` economiza memória de ativações recomputando parte delas durante o backward, em troca de mais tempo de processamento. Reduzir `CONTEXT_WINDOW` também pode ajudar. Veja os guias sobre [acumulação de gradientes](./docs/gradient-accumulation.md) e [gradient checkpointing](./docs/gradient-checkpointing.md).

---

## ▶ Como Executar Localmente

### Opção 1: Usando o script principal (Recomendado)
```bash
python run.py
```

### Opção 2: Usando o Uvicorn diretamente
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --env-file .env.local
```

Após iniciar, o servidor estará acessível em: **`http://localhost:8000`** ou **`http://0.0.0.0:8000`**.

---

## 🐳 Execução com Docker e Docker Compose (Windows/Linux)

O container é Linux tanto no Docker Desktop/WSL2 quanto no Linux nativo. A raiz do
projeto é montada diretamente em `/workspace`: checkpoint, métricas, datasets e
artefatos são os mesmos arquivos do host. Renomear ou substituir uma pasta de modelo
no Windows/Linux passa a valer imediatamente no container; não existe cópia interna
do checkpoint.

### 1. Pré-requisitos

- Windows: Docker Desktop com backend WSL2, integração NVIDIA e containers Linux.
- Linux: Docker Engine, Compose v2 e `nvidia-container-toolkit`.

No Linux, instale o toolkit se necessário:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Teste o acesso da GPU pelo Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

### 2. Subir a Aplicação com Docker Compose

Na raiz do projeto, execute:

```bash
# Constrói os runtimes nativos, instala CUDA/PyTorch e inicia a aplicação
docker compose up --build -d

# Visualizar logs em tempo real
docker compose logs -f
```

Acesse `http://localhost:8000`. O Compose lê `.env` e `.env.local` diretamente
do host; esses arquivos, os modelos, checkpoints, datasets e artefatos são ignorados
pelo Git e pelo contexto de build.

Os caminhos podem ficar vazios:

```env
CHECKPOINT_DIR=
FINE_TUNING_SOURCE_DIR=
FINE_TUNING_ROOT_DIR=
```

Nesse caso, o pré-treino usa sempre `checkpoints/base`. O SFT herda o checkpoint efetivo do pré-treino e
grava por padrão em `checkpoints_finetuning/base_finetuning`. Se
`FINE_TUNING_ROOT_DIR` for definido, a compilação lê `base_finetuning` dentro desse
diretório escolhido.

> Execute somente um escritor por vez. Não mantenha um Python do Windows treinando
> simultaneamente com o container. O projeto também usa um lock compartilhado para
> bloquear pré-treino e SFT concorrentes.

### 3. Parar o Container
```bash
docker compose down
```

O container recebe até dois minutos para concluir o batch e salvar ao ser encerrado.
Para acompanhar um treino retomado:

```bash
docker compose logs -f app
```

### 4. Executar em modo CPU (sem GPU NVIDIA no Docker)
Caso vá executar o container em um servidor sem GPU dedicada:
1. Comente ou remova o bloco `deploy.resources.reservations.devices` no `docker-compose.yml`.
2. Configure `INFERENCE_DEVICE=cpu` e `FINE_TUNING_DEVICE=cpu` no seu `.env.local`.
3. Inicie com `docker compose up -d`.

---

## 🖥 Mapeamento das Interfaces Web

A aplicação disponibiliza interfaces prontas e reativas acessíveis diretamente pelo navegador:

| Rota | Descrição |
| :--- | :--- |
| **`/`** | **Dashboard de Pré-Treino**: Controle de início/pausa/retomada de treinamento, telemetria de perda em tempo real via WebSockets, gradientes e status de hardware. |
| **`/finetuning`** | **Supervised Fine-Tuning (SFT)**: Dashboard de fine-tuning conversacional, histórico de métricas de loss/acurácia e playground para testar snapshots do checkpoint. |
| **`/compile`** | **Compilação e exportação**: verificações de ambiente e conversão do checkpoint para GGUF usando um runtime compatível com a arquitetura. |
| **`/playground`** | **Playground de Inferência**: Interface de chat dedicada para teste em tempo real de modelos e artefatos compilados em CPU ou GPU. |
| **`/docs`** | **Swagger UI**: Documentação interativa de todas as rotas REST e endpoints da API FastAPI. |
| **`/redoc`** | **ReDoc**: Documentação técnica detalhada da API. |

> **Nota sobre o fluxo de Checkpoints**: O primeiro fine-tuning copia automaticamente os pesos e a arquitetura de `checkpoints/base` para `checkpoints_finetuning/base_finetuning`. O otimizador AdamW do SFT opera com estado isolado salvo nessa pasta, permitindo pausas e retomadas sem interferir no pré-treino.
>
> O modo LoRA usa `checkpoints_finetuning/base_finetuning_lora`, preserva o modelo-base em
> `base_model.safetensors`, salva o adaptador em `adapter_model.safetensors` e publica também
> `model.safetensor` mesclado para compatibilidade. As duas pastas permanecem no mesmo bind mount
> compartilhado entre Windows e Docker.

---

## 📂 Estrutura do Projeto

```text
AI-Training-Studio/
├── Dockerfile                   # Build conteinerizado com suporte a CUDA 12.4
├── docker-compose.yml           # Orquestração com GPU passthrough e volumes
├── requirements.txt             # Dependências Python essenciais
├── run.py                       # Entrypoint padrão da aplicação FastAPI
├── readmme.md                   # Resumo e histórico original mantido
├── README.md                    # Manual completo e documentação oficial
├── .env.local                   # Variáveis de ambiente e hiperparâmetros
├── src/
│   ├── main.py                  # Servidor FastAPI e rotas HTTP/WebSocket
│   ├── config.py                # Configurações de pré-treino via Pydantic
│   ├── model.py                 # Arquitetura customizada do modelo LLM
│   ├── train.py                 # Loop de pré-treino e rotinas de salvamento
│   ├── tokenizer.py             # Instanciação do tokenizador
│   ├── base_inference.py        # Módulo de inferência de checkpoints base
│   ├── compile_runtime/         # Conversor de GGUF e serviço de runtime C++
│   └── finetuning/              # Módulo de Supervised Fine-Tuning (SFT)
├── templates/                   # Interfaces HTML das páginas web
├── static/                      # Arquivos estáticos (CSS, JS, assets)
├── runtime/                     # Código-fonte e scripts de build dos runtimes nativos
├── checkpoints/                 # Armazenamento de checkpoints de pré-treino
└── checkpoints_finetuning/      # Armazenamento de checkpoints de SFT
```

---

## 🔧 Solução de Problemas (Troubleshooting)

### 1. `CUDA out of memory` (OOM)
- Diminua `TRAIN_BATCH_SIZE` ou `FINE_TUNING_MICRO_BATCH_SIZE` no `.env.local` (ex: de 6 para 2 ou 4).
- Aumente `GRADIENT_ACCUMULATION_STEPS` proporcionalmente para manter o tamanho global do lote constante.
- Reduza a `CONTEXT_WINDOW` (ex: de 512 para 256).

### 2. Erro de carregamento do BitsAndBytes no Linux
- Certifique-se de que a variável de ambiente `LD_LIBRARY_PATH` aponta para o diretório de bibliotecas do CUDA se estiver rodando bare-metal:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```

### 3. Permissão negada ao salvar checkpoints no Docker
- Se estiver rodando no Linux com Docker, verifique as permissões das pastas de volume:
  ```bash
  mkdir -p checkpoints checkpoints_finetuning data artifacts
  chmod -R 777 checkpoints checkpoints_finetuning data artifacts
  ```

### 4. Erro ao baixar Datasets ou Tokenizers do Hugging Face
- Verifique se a chave `HF_TOKEN` no seu `.env.local` possui permissão de leitura (Read Token).
