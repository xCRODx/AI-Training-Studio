# RetBit runtime core

This directory is the isolated C++ inference backend for the project's
`CustomLLM / RetNet / BitLinear` architecture. It contains:

- a stateful recurrent RetNet retention operator equivalent to the causal
  parallel PyTorch implementation;
- a packed 2-bit CPU operator for ternary `BitLinear` weights (four logical
  values from `{-1, 0, 1}` per byte, with one FP32 scale per matrix);
- numerical tests for parallel/recurrent equivalence and token-by-token state.
- a GGUF v3 loader and the complete token-by-token graph: embedding, RMSNorm,
  Q/K/V retention, output projection, SiLU FFN, tied output head and sampling;
- `retbit_cli`, used by the compiled Playground without loading PyTorch.
- a persistent server mode that loads GGUF once and streams token IDs as soon
  as they are sampled;
- a fixed CPU worker pool used by ternary projections and the tied output head.

Build and run the numerical tests on Windows with the installed Clang toolchain:

```powershell
powershell -ExecutionPolicy Bypass -File runtime/retbit/build_windows.ps1
```

The corresponding converter is `src/compile_runtime/converter.py`. Tokenization
and detokenization remain in the Python API process using the exact checkpoint
SentencePiece tokenizer; all neural-network inference runs in the C++ process.
The Playground also exposes an explicitly labelled CUDA alternative backed by
the persistent BF16 PyTorch checkpoint; it is not presented as a C++ CUDA
kernel. CPU thread count is selectable and changing it safely rotates to a
versioned runtime worker.
