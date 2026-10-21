from __future__ import annotations

import ctypes
from pathlib import Path

import torch


class RetBitCudaRuntime:
    def __init__(self) -> None:
        self._library: ctypes.CDLL | None = None
        self._error: str | None = None

    @property
    def path(self) -> Path:
        deployed = sorted(
            Path("artifacts/retbit-cuda").resolve().glob("retbit_cuda_sm120-*.dll"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if deployed:
            return deployed[0]
        return Path("artifacts/retbit-cuda/retbit_cuda_sm120.dll").resolve()

    @property
    def available(self) -> bool:
        if self._library is not None:
            return True
        if self._error is not None or not self.path.is_file() or not torch.cuda.is_available():
            return False
        try:
            library = ctypes.CDLL(str(self.path))
            library.retbit_ternary_linear_bf16.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            library.retbit_ternary_linear_bf16.restype = ctypes.c_int
            library.retbit_cuda_error_string.argtypes = [ctypes.c_int]
            library.retbit_cuda_error_string.restype = ctypes.c_char_p
            library.retbit_sample_topk_bf16.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            library.retbit_sample_topk_bf16.restype = ctypes.c_int
            self._library = library
            return True
        except OSError as exc:
            self._error = str(exc)
            return False

    @property
    def error(self) -> str | None:
        return self._error

    def linear(
        self,
        input_tensor: torch.Tensor,
        packed_weight: torch.Tensor,
        scale: float,
        output_features: int,
    ) -> torch.Tensor:
        if not self.available or self._library is None:
            raise RuntimeError(self._error or "O kernel RetBit CUDA não está disponível")
        if input_tensor.dtype != torch.bfloat16 or not input_tensor.is_cuda:
            raise TypeError("RetBit CUDA requires a CUDA BF16 input")
        source = input_tensor.contiguous()
        rows = source.numel() // source.shape[-1]
        output = torch.empty((*source.shape[:-1], output_features), dtype=source.dtype, device=source.device)
        stream = torch.cuda.current_stream(source.device).cuda_stream
        code = self._library.retbit_ternary_linear_bf16(
            ctypes.c_void_p(source.data_ptr()),
            ctypes.c_void_p(packed_weight.data_ptr()),
            ctypes.c_float(scale),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(rows),
            ctypes.c_int(source.shape[-1]),
            ctypes.c_int(output_features),
            ctypes.c_void_p(stream),
        )
        if code:
            message = self._library.retbit_cuda_error_string(code)
            raise RuntimeError(message.decode("utf-8") if message else f"CUDA error {code}")
        return output

    def sample_topk(
        self,
        logits: torch.Tensor,
        workspace: torch.Tensor,
        repetition_mask: torch.Tensor,
        output: torch.Tensor,
        repetition_penalty: float,
        temperature: float,
        top_p: float,
        top_k: int,
        uniform: float,
    ) -> torch.Tensor:
        if not self.available or self._library is None:
            raise RuntimeError(self._error or "O kernel RetBit CUDA não está disponível")
        stream = torch.cuda.current_stream(logits.device).cuda_stream
        code = self._library.retbit_sample_topk_bf16(
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_void_p(repetition_mask.data_ptr()),
            ctypes.c_float(repetition_penalty),
            ctypes.c_float(temperature),
            ctypes.c_float(top_p),
            ctypes.c_int(top_k),
            ctypes.c_float(uniform),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_int(logits.shape[-1]),
            ctypes.c_void_p(stream),
        )
        if code:
            message = self._library.retbit_cuda_error_string(code)
            raise RuntimeError(message.decode("utf-8") if message else f"CUDA error {code}")
        return output


retbit_cuda_runtime = RetBitCudaRuntime()
