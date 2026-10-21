# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_REF=6fdd0ac8907fd973a42b876357823ad2124cd8ed
ARG BITNET_CPP_REF=0b341e582afbf9e1011f24744b554c96a3477eb5

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    PYTHONPATH=/workspace \
    HF_HOME=/workspace/.cache/huggingface \
    LLAMA_CPP_SERVER=/opt/retbit-runtimes/llama-server \
    LLAMA_CPP_CLI=/opt/retbit-runtimes/llama-cli \
    BITNET_CPP_SERVER=/opt/retbit-runtimes/bitnet-server \
    BITNET_CPP_CLI=/opt/retbit-runtimes/bitnet-cli \
    RETBIT_CPP_CLI=/opt/retbit-runtimes/retbit-cli

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git libcurl4-openssl-dev \
        gnupg libgomp1 libopenblas-dev lsb-release ninja-build pkg-config python3 \
        python3-dev python3-pip python3-venv software-properties-common wget \
    && wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh \
    && chmod +x /tmp/llvm.sh \
    && /tmp/llvm.sh 18 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -f /tmp/llvm.sh \
    && rm -rf /var/lib/apt/lists/*

# Runtimes externos fixados por commit para builds reproduzíveis. São builds CPU
# portáveis; o treino e a inferência PyTorch continuam usando CUDA no container.
RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git /opt/src/llama.cpp \
    && git -C /opt/src/llama.cpp checkout "${LLAMA_CPP_REF}" \
    && cmake -S /opt/src/llama.cpp -B /opt/src/llama.cpp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
        -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
    && cmake --build /opt/src/llama.cpp/build --target llama-server llama-cli -j 2 \
    && mkdir -p /opt/retbit-runtimes \
    && cp /opt/src/llama.cpp/build/bin/llama-server /opt/retbit-runtimes/llama-server \
    && cp /opt/src/llama.cpp/build/bin/llama-cli /opt/retbit-runtimes/llama-cli \
    && rm -rf /opt/src/llama.cpp

# O fork BitNet requer Clang e usa o runtime OpenMP correspondente. Esta camada
# fica separada para preservar o cache do build llama.cpp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libomp-18-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/microsoft/BitNet.git /opt/src/BitNet \
    && git -C /opt/src/BitNet checkout "${BITNET_CPP_REF}" \
    && git -C /opt/src/BitNet submodule update --init --recursive \
    && cmake -S /opt/src/BitNet -B /opt/src/BitNet/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DBITNET_X86_TL2=OFF \
        -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON \
    && cmake --build /opt/src/BitNet/build --target llama-server llama-cli -j 2 \
    && cp /opt/src/BitNet/build/bin/llama-server /opt/retbit-runtimes/bitnet-server \
    && cp /opt/src/BitNet/build/bin/llama-cli /opt/retbit-runtimes/bitnet-cli \
    && rm -rf /opt/src/BitNet

WORKDIR /opt/retbit-app
COPY requirements.txt ./requirements.txt
RUN python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install -r requirements.txt

# Uma cópia mínima permite diagnóstico sem volume. Em execução, o Compose monta
# o diretório real do host em /workspace: Windows e Linux leem os mesmos arquivos.
COPY . /opt/retbit-app
RUN cmake -S /opt/retbit-app/runtime/retbit -B /tmp/retbit-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/retbit-build --target retbit_cli_build -j 2 \
    && cp /tmp/retbit-build/retbit_cli_build /opt/retbit-runtimes/retbit-cli \
    && rm -rf /tmp/retbit-build

COPY docker/entrypoint.sh /usr/local/bin/retbit-entrypoint
RUN chmod +x /usr/local/bin/retbit-entrypoint

WORKDIR /workspace
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/retbit-entrypoint"]
CMD ["python", "run.py"]
