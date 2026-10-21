from __future__ import annotations

import os
import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import atexit
from pathlib import Path
from typing import Generator

from src.finetuning.config import fine_tuning_settings
from src.finetuning.inference import FineTuningInferenceService
from src.finetuning.chat_format import format_generation_prompt
from src.tokenizer import decode_continuation, tokenizer


class PersistentRetBitService:
    def __init__(self):
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._signature: tuple[str, int, str, int] | None = None

    @property
    def executable(self) -> Path:
        configured = os.getenv("RETBIT_CPP_CLI", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        deployed = sorted(
            list(Path("artifacts/retbit-runtime").resolve().glob("retbit_cli-*")),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if deployed:
            return deployed[0]
        candidates = (
            Path("runtime/retbit/build-windows/retbit_cli_build.exe"),
            Path("runtime/retbit/build/retbit_cli"),
        )
        return next((path.resolve() for path in candidates if path.is_file()), candidates[0].resolve())

    @property
    def loaded(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        return format_generation_prompt(messages, fine_tuning_settings.system_prompt)

    def _context_window(self) -> int:
        config_path = fine_tuning_settings.base_path / "config.json"
        if not config_path.is_file():
            raise RuntimeError("O checkpoint SFT não registra o contexto efetivo do artefato.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        context = int(config.get("context_window", 0) or 0)
        if context <= 1:
            raise RuntimeError("context_window inválido no checkpoint SFT.")
        return context

    def _stop_owned_process(self) -> None:
        process = self._process
        self._process = None
        self._signature = None
        if process is None or process.poll() is not None:
            return
        if process.stdin:
            process.stdin.write("QUIT\n")
            process.stdin.flush()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("O worker RetBit anterior não encerrou de forma segura.") from exc

    def _ensure_process(self, artifact: Path, threads: int) -> subprocess.Popen[str]:
        executable = self.executable
        if not executable.is_file():
            raise RuntimeError("O executável RetBit ainda não foi compilado.")
        if not artifact.is_file():
            raise RuntimeError("O artefato GGUF selecionado não existe.")
        threads = max(1, min(int(threads), 128))
        signature = (str(artifact.resolve()), artifact.stat().st_mtime_ns, str(executable), threads)
        if self.loaded and signature == self._signature:
            return self._process  # type: ignore[return-value]
        self._stop_owned_process()
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            [str(executable), "--model", str(artifact), "--server", "--threads", str(threads)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creation_flags,
        )
        ready = process.stdout.readline().strip() if process.stdout else ""
        if not ready.startswith("READY:"):
            error = process.stderr.read().strip() if process.stderr else ready
            raise RuntimeError(f"O worker RetBit não inicializou: {error}")
        self._process = process
        self._signature = signature
        return process

    def preload(self, artifact: Path, threads: int = 12) -> dict[str, str | int | bool | float]:
        with self._lock:
            started = time.perf_counter()
            process = self._ensure_process(artifact, threads)
            return {
                "loaded": True,
                "device": "cpu",
                "runtime": "retbit-cpp-persistent",
                "pid": process.pid,
                "load_seconds": round(time.perf_counter() - started, 3),
                "artifact": artifact.name,
                "threads": max(1, min(int(threads), 128)),
            }

    def stream(
        self,
        artifact: Path,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: int,
        threads: int = 12,
    ) -> Generator[dict[str, object], None, None]:
        with self._lock:
            process = self._ensure_process(artifact, threads)
            prompt = self.format_prompt(messages)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            context = self._context_window()
            input_ids = input_ids[-context:]
            request = "|".join([
                ",".join(str(token) for token in input_ids), str(min(max_new_tokens, context)),
                str(temperature), str(top_p), str(top_k), str(repetition_penalty), str(seed),
                str(tokenizer.eos_token_id),
            ])
            if not process.stdin or not process.stdout:
                raise RuntimeError("Pipes do worker RetBit indisponíveis.")
            process.stdin.write(request + "\n")
            process.stdin.flush()
            generated: list[int] = []
            decoded_so_far = ""
            while True:
                line = process.stdout.readline()
                if not line:
                    error = process.stderr.read().strip() if process.stderr else "worker finalizado"
                    raise RuntimeError(f"O worker RetBit parou: {error}")
                line = line.strip()
                if line.startswith("TOKEN:"):
                    token_id = int(line.removeprefix("TOKEN:"))
                    generated.append(token_id)
                    decoded = decode_continuation(input_ids, generated)
                    delta = decoded[len(decoded_so_far):] if decoded.startswith(decoded_so_far) else decoded
                    decoded_so_far = decoded
                    if delta:
                        yield {"type": "token", "text": delta, "token_id": token_id}
                elif line.startswith("DONE:"):
                    _, count, milliseconds = line.split(":", 2)
                    elapsed = max(int(milliseconds) / 1000, 0.001)
                    evaluated = len(input_ids) + int(count)
                    yield {
                        "type": "done", "token_count": int(count),
                        "prompt_token_count": len(input_ids), "evaluated_token_count": evaluated,
                        "elapsed_seconds": round(elapsed, 3),
                        "tokens_per_second": round(int(count) / elapsed, 2),
                        "evaluated_tokens_per_second": round(evaluated / elapsed, 2),
                        "runtime": "retbit-cpp-persistent", "device": "cpu",
                        "threads": max(1, min(int(threads), 128)),
                    }
                    return
                elif line.startswith("ERROR:"):
                    raise RuntimeError(line.removeprefix("ERROR:"))


compiled_service = PersistentRetBitService()
gpu_settings = fine_tuning_settings.model_copy(update={"chat_device": "cuda"})
gpu_inference_service = FineTuningInferenceService(gpu_settings)


class NativeCppService:
    """Persistent llama-server compatible runtime with a real KV cache."""

    def __init__(self, target: str, env_name: str, command_name: str, port: int):
        self.target = target
        self.env_name = env_name
        self.command_name = command_name
        self.port = port
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._signature: tuple[str, int, int] | None = None
        self._log_handle = None

    @property
    def executable(self) -> Path:
        configured = os.getenv(self.env_name, "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        discovered = shutil.which(self.command_name)
        if discovered:
            return Path(discovered).resolve()
        windows_name = f"{self.command_name}.exe"
        local_candidates = (
            Path("runtime") / self.target / "build" / "bin" / self.command_name,
            Path("runtime") / self.target / "build" / "bin" / windows_name,
        )
        return next(
            (candidate.resolve() for candidate in local_candidates if candidate.is_file()),
            local_candidates[0].resolve(),
        )

    @property
    def loaded(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._signature = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _ensure_process(self, artifact: Path, threads: int) -> subprocess.Popen[bytes]:
        executable = self.executable
        if not executable.is_file():
            raise RuntimeError(
                f"{self.target} não está instalado. Use o Dockerfile do projeto ou defina {self.env_name}."
            )
        if not artifact.is_file():
            raise RuntimeError("O artefato GGUF selecionado não existe.")
        threads = max(1, min(int(threads), 128))
        signature = (str(artifact.resolve()), artifact.stat().st_mtime_ns, threads)
        if self.loaded and self._signature == signature and self._health():
            return self._process  # type: ignore[return-value]

        self.stop()
        log_dir = Path("artifacts/runtime-logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.target.replace('.', '-')}.log"
        self._log_handle = log_path.open("ab", buffering=0)
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            [
                str(executable),
                "--model", str(artifact.resolve()),
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--threads", str(threads),
                "--n-gpu-layers", "0",
            ],
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        self._signature = signature
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                try:
                    detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                except OSError:
                    detail = "sem log"
                self.stop()
                raise RuntimeError(f"{self.target} recusou o GGUF: {detail.strip()}")
            if self._health():
                return self._process
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"{self.target} não ficou pronto em 180 segundos.")

    def preload(self, artifact: Path, threads: int = 12) -> dict[str, str | int | bool | float]:
        with self._lock:
            started = time.perf_counter()
            process = self._ensure_process(artifact, threads)
            return {
                "loaded": True,
                "device": "cpu",
                "runtime": self.target,
                "pid": process.pid,
                "load_seconds": round(time.perf_counter() - started, 3),
                "artifact": artifact.name,
                "threads": max(1, min(int(threads), 128)),
                "kv_cache": {"enabled": True, "implementation": "llama-server", "cache_prompt": True},
            }

    def stream(
        self,
        artifact: Path,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: int,
        threads: int = 12,
    ) -> Generator[dict[str, object], None, None]:
        with self._lock:
            self._ensure_process(artifact, threads)
            prompt = format_generation_prompt(messages, fine_tuning_settings.system_prompt)
            payload = json.dumps(
                {
                    "prompt": prompt,
                    "n_predict": max(1, int(max_new_tokens)),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "repeat_penalty": float(repetition_penalty),
                    "seed": int(seed),
                    "stream": True,
                    "cache_prompt": True,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/completion",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            token_count = 0
            with urllib.request.urlopen(request, timeout=600) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    content = str(event.get("content", ""))
                    if content:
                        token_count += 1
                        yield {"type": "token", "text": content}
                    if event.get("stop"):
                        timings = event.get("timings") or {}
                        predicted = int(timings.get("predicted_n", token_count) or token_count)
                        elapsed = max(time.perf_counter() - started, 0.001)
                        yield {
                            "type": "done",
                            "token_count": predicted,
                            "elapsed_seconds": round(elapsed, 3),
                            "tokens_per_second": round(
                                float(timings.get("predicted_per_second", predicted / elapsed)), 2
                            ),
                            "runtime": self.target,
                            "device": "cpu",
                            "kv_cache": True,
                        }
                        return


llama_cpp_service = NativeCppService("llama.cpp", "LLAMA_CPP_SERVER", "llama-server", 18180)
bitnet_cpp_service = NativeCppService("bitnet.cpp", "BITNET_CPP_SERVER", "bitnet-server", 18181)


def native_service_for_target(target: str) -> NativeCppService | None:
    if target == "llama.cpp":
        return llama_cpp_service
    if target == "bitnet.cpp":
        return bitnet_cpp_service
    return None


atexit.register(llama_cpp_service.stop)
atexit.register(bitnet_cpp_service.stop)
