import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import torch

from src.finetuning.config import fine_tuning_settings
from src.config import settings as pretrain_settings
from src.finetuning.inference import FineTuningInferenceService
from src.finetuning.manager import finetuning_manager
from src.compile_runtime.converter import convert_checkpoint_to_gguf
from src.compile_runtime.service import (
    bitnet_cpp_service,
    compiled_service,
    gpu_inference_service,
    llama_cpp_service,
    native_service_for_target,
)
from src.tokenizer import tokenizer
from src.sampling import SamplingParameters, resolve_sampling
from src.lifecycle import shutdown_requested

router = APIRouter()
inference_service = FineTuningInferenceService(fine_tuning_settings)


def _template(name: str) -> str:
    return (Path("templates") / name).read_text(encoding="utf-8")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


class FineTuningChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = Field(default=64, ge=1)
    sampling_mode: Literal["auto", "greedy", "sample"] = "auto"
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=1, ge=0, le=1000)
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=2)
    seed: int | None = None


class FineTuningStartRequest(BaseModel):
    method: Literal["full", "lora"] = "full"
    lora_rank: int = Field(default=16, ge=1, le=1024)
    lora_alpha: float = Field(default=32.0, gt=0, le=8192)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: str = Field(
        default="q_proj,k_proj,v_proj,out_proj,g_proj,gate_proj,up_proj,down_proj,fc1,fc2",
        min_length=1,
        max_length=1000,
    )


class CompiledChatRequest(FineTuningChatRequest):
    artifact: str = Field(min_length=1, max_length=255)
    device: Literal["cpu", "cuda"] = "cpu"
    threads: int = Field(default=12, ge=1, le=128)


class CompiledLoadRequest(BaseModel):
    artifact: str = Field(min_length=1, max_length=255)
    device: Literal["cpu", "cuda"] = "cpu"
    threads: int = Field(default=12, ge=1, le=128)


def _resolve_request_sampling(request: FineTuningChatRequest) -> SamplingParameters:
    return resolve_sampling(
        request.sampling_mode,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        auto_temperature=pretrain_settings.inference_auto_temperature,
        auto_top_p=pretrain_settings.inference_auto_top_p,
        auto_top_k=pretrain_settings.inference_auto_top_k,
        auto_repetition_penalty=pretrain_settings.inference_auto_repetition_penalty,
    )


@router.get("/finetuning", response_class=HTMLResponse)
async def finetuning_page():
    return _template("finetuning.html")


@router.get("/compile", response_class=HTMLResponse)
async def compile_page():
    return _template("compile.html")


@router.get("/playground", response_class=HTMLResponse)
async def playground_page():
    return _template("playground.html")


@router.get("/fundamentos", response_class=HTMLResponse)
async def fundamentos_page():
    return _template("fundamentos.html")


@router.get("/api/finetuning/status")
async def finetuning_status():
    return finetuning_manager.status()


@router.get("/api/finetuning/metrics/history")
async def finetuning_metrics_history(limit: int = 300):
    return {"items": finetuning_manager.history(max(1, min(limit, 1000)))}


@router.post("/api/finetuning/start")
async def finetuning_start(request: FineTuningStartRequest | None = None):
    try:
        options = request.model_dump() if request is not None else {}
        return await asyncio.to_thread(finetuning_manager.start, **options)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/finetuning/pause")
async def finetuning_pause():
    try:
        return finetuning_manager.request_pause()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/finetuning/resume")
async def finetuning_resume():
    try:
        return await asyncio.to_thread(finetuning_manager.resume)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/finetuning/stop")
async def finetuning_stop():
    try:
        return finetuning_manager.request_stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/finetuning/checkpoint")
async def finetuning_checkpoint():
    try:
        return finetuning_manager.request_checkpoint()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/finetuning/chat")
async def finetuning_chat(request: FineTuningChatRequest):
    if not finetuning_manager.status()["model"]["ready"]:
        raise HTTPException(status_code=409, detail="Inicie o fine-tuning para criar base_finetuning.")
    messages = [message.model_dump() for message in request.messages]
    inference_service.set_settings(finetuning_manager.settings)
    try:
        sampling = _resolve_request_sampling(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def stream():
        try:
            yield from inference_service.generate(
                messages=messages,
                max_new_tokens=request.max_new_tokens,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                repetition_penalty=sampling.repetition_penalty,
                seed=request.seed,
            )
        except Exception as exc:
            yield f"[Erro: {exc}]"

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


@router.websocket("/ws/finetuning/metrics")
async def finetuning_metrics_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        while not shutdown_requested.is_set():
            await websocket.send_json(finetuning_manager.status())
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return


def _compiled_artifacts() -> list[dict[str, str | int | float]]:
    roots = [Path("artifacts/bitnet").resolve(), Path("artifacts/llama").resolve()]
    artifacts = []
    for root in roots:
        if not root.exists():
            continue
        for model in root.rglob("*.gguf"):
            artifacts.append(
                {
                    "name": model.stem,
                    "path": str(model),
                    "size_mb": round(model.stat().st_size / 1024**2, 2),
                    "modified_at": model.stat().st_mtime,
                    "target": _artifact_target(model.stem),
                }
            )
    return sorted(artifacts, key=lambda item: float(item["modified_at"]), reverse=True)


def _artifact_target(stem: str) -> str:
    low = stem.lower()
    if "retbit" in low:
        return "retbit-cpp"
    if "llama" in low:
        return "llama.cpp"
    return "bitnet.cpp"


def _compiled_architecture() -> tuple[str, str]:
    """(architecture, target) do modelo em base_finetuning (fonte da compilação)."""
    arch_info = finetuning_manager.status()["architecture"]
    architecture = arch_info.get("architecture") or "transformer"
    if architecture == "retbit":
        return architecture, "retbit-cpp"
    if architecture == "retnet":
        return architecture, "retnet-float-unsupported"
    if architecture == "bitnet":
        return architecture, "bitnet.cpp"
    return architecture or "transformer", "llama.cpp"


def _tool_info(name: str, known_paths: tuple[Path, ...] = ()) -> dict[str, str | bool]:
    """Find build tools even when Uvicorn predates a Windows PATH update."""
    discovered = shutil.which(name)
    executable = Path(discovered) if discovered else next(
        (path for path in known_paths if path.is_file()),
        None,
    )
    if executable is None:
        return {"available": False, "path": "", "version": ""}

    version = ""
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        version = (result.stdout or result.stderr).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"available": True, "path": str(executable), "version": version}


def _retbit_core_info() -> dict[str, str | bool]:
    executable = Path("runtime/retbit/build-windows/retbit_core_tests.exe").resolve()
    if not executable.is_file():
        return {"available": False, "path": str(executable), "version": "Build pendente"}
    try:
        result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        passed = result.returncode == 0
        detail = (result.stdout or result.stderr).strip().splitlines()
        return {
            "available": passed,
            "path": str(executable),
            "version": detail[-1] if detail else ("OK" if passed else "Teste falhou"),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "path": str(executable), "version": str(exc)}


def _retbit_runtime_info() -> dict[str, str | bool]:
    executable = compiled_service.executable
    return {
        "available": executable.is_file(),
        "path": str(executable),
        "version": "GGUF v3 loader + RetNet graph" if executable.is_file() else "Build pendente",
    }


def _native_runtime_info(target: str) -> dict[str, str | bool]:
    service = native_service_for_target(target)
    if service is None:
        return {"available": False, "path": "", "version": "Runtime desconhecido"}
    executable = service.executable
    version = "Instalado" if executable.is_file() else "Use docker compose up --build"
    if executable.is_file():
        try:
            result = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            first_line = (result.stdout or result.stderr).strip().splitlines()
            if first_line:
                version = first_line[0]
        except (OSError, subprocess.SubprocessError):
            pass
    return {"available": executable.is_file(), "path": str(executable), "version": version}


@router.get("/api/compile/preflight")
async def compile_preflight():
    from src.architecture import architecture_label

    model = finetuning_manager.status()["model"]
    architecture, target = _compiled_architecture()
    uses_custom_runtime = target == "retbit-cpp"
    cmake = _tool_info(
        "cmake",
        (Path("C:/Program Files/CMake/bin/cmake.exe"),),
    )
    clang = _tool_info(
        "clang",
        (
            Path("C:/Program Files/LLVM/bin/clang.exe"),
            Path("C:/Program Files (x86)/LLVM/bin/clang.exe"),
        ),
    )
    retbit_core = await asyncio.to_thread(_retbit_core_info)
    retbit_runtime = _retbit_runtime_info()
    native_runtime = _native_runtime_info(target)

    if architecture == "retnet":
        checks = {
            "fine_tuning_model": bool(model.get("valid")),
            "cmake": True,
            "clang": True,
            "retnet_core": True,
            "architecture_supported": False,
        }
        ready = False
        blocker = (
            "O runtime C++ disponível utiliza pesos ternários e não oferece suporte exato a projeções RetNet em ponto flutuante. "
            "A compilação foi bloqueada para não mudar os pesos silenciosamente."
        )
    elif uses_custom_runtime:
        checks = {
            "fine_tuning_model": bool(model.get("valid")),
            "cmake": bool(cmake["available"]),
            "clang": bool(clang["available"]),
            "retnet_core": bool(retbit_core["available"]),
            "architecture_supported": bool(retbit_runtime["available"]),
        }
        ready = all(checks.values())
        blocker = "Todos os componentes do runtime C++ estão disponíveis."
    else:
        # A exportação é Python puro, mas só é declarada pronta quando o runtime
        # Linux/Windows correspondente está disponível para validar o GGUF.
        checks = {
            "fine_tuning_model": bool(model.get("valid")),
            "cmake": True,
            "clang": True,
            "retnet_core": True,
            "architecture_supported": bool(native_runtime["available"]),
        }
        ready = all(checks.values())
        if ready:
            blocker = f"Exportação e execução {target} disponíveis no Playground."
        elif not checks["fine_tuning_model"]:
            blocker = (
                "O runtime está disponível, mas ainda não existe um base_finetuning válido em "
                f"{model.get('path', fine_tuning_settings.base_path)}."
            )
        else:
            blocker = (
                f"O runtime {target} não está instalado neste processo. "
                "Use docker compose up --build."
            )

    check_details = {
        "cmake": cmake,
        "clang": clang,
        "retnet_core": retbit_core,
        "architecture_supported": retbit_runtime if uses_custom_runtime else native_runtime,
    }
    return {
        "checks": checks,
        "check_details": check_details,
        "ready": ready,
        "model": model,
        "architecture": architecture,
        "architecture_label": architecture_label(architecture),
        "target": target,
        "uses_custom_runtime": uses_custom_runtime,
        "kv_cache": {
            "enabled": architecture in {"transformer", "bitnet"},
            "stage": "runtime",
            "implementation": target if architecture in {"transformer", "bitnet"} else "recurrent-state",
        },
        "blocker": blocker,
        "artifacts": _compiled_artifacts(),
    }


@router.post("/api/compile/start")
async def compile_start():
    preflight = await compile_preflight()
    if not preflight["model"].get("valid"):
        raise HTTPException(status_code=409, detail="Não existe um base_finetuning válido para compilar.")
    if not preflight["ready"]:
        raise HTTPException(status_code=409, detail=preflight["blocker"])
    if finetuning_manager.is_active():
        raise HTTPException(status_code=409, detail="Pause o fine-tuning antes de publicar um artefato compilado.")

    architecture, target = _compiled_architecture()
    if architecture == "retbit":
        filename = "base_finetuning-retbit.gguf"
        folder = Path("artifacts/bitnet")
    elif architecture == "retnet":
        raise HTTPException(
            status_code=409,
            detail="Não há um runtime compilado compatível com projeções RetNet em ponto flutuante; a conversão foi bloqueada.",
        )
    elif architecture == "bitnet":
        filename = "base_finetuning-bitnet.gguf"
        folder = Path("artifacts/bitnet")
    else:
        filename = "base_finetuning-llama.gguf"
        folder = Path("artifacts/llama")
    destination = (folder / filename).resolve()
    try:
        artifact = await asyncio.to_thread(
            convert_checkpoint_to_gguf,
            finetuning_manager.settings.base_path,
            destination,
            architecture=architecture,
        )
        if target == "retbit-cpp":
            validation = await asyncio.to_thread(
                subprocess.run,
                [str(compiled_service.executable), "--model", str(destination), "--validate"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if validation.returncode != 0:
                raise RuntimeError((validation.stderr or validation.stdout).strip())
            message = validation.stdout.strip()
        else:
            service = native_service_for_target(target)
            if service is None:
                raise RuntimeError(f"Runtime desconhecido para o alvo {target}.")
            validation = await asyncio.to_thread(service.preload, destination, 2)
            message = (
                f"Artefato GGUF exportado e carregado com sucesso no {target}; "
                f"KV cache ativo (PID {validation['pid']})."
            )
        return {"status": "success", "message": message, "artifact": artifact}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha na compilação ({architecture}): {exc}") from exc


@router.get("/api/playground/status")
async def playground_status():
    artifacts = _compiled_artifacts()
    has_retbit_artifact = any(item["target"] == "retbit-cpp" for item in artifacts)
    has_native_cpu_artifact = any(
        (service := native_service_for_target(str(item["target"]))) is not None
        and service.executable.is_file()
        for item in artifacts
    )
    return {
        "ready": bool(artifacts),
        "artifacts": artifacts,
        "devices": {
            "cpu": (has_retbit_artifact and compiled_service.executable.is_file()) or has_native_cpu_artifact,
            "cuda": torch.cuda.is_available(),
        },
        "loaded": {
            "cpu": compiled_service.loaded or llama_cpp_service.loaded or bitnet_cpp_service.loaded,
            "cuda": gpu_inference_service.loaded,
        },
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "gpu_runtime": "CUDA Graph" if torch.cuda.is_available() else "",
        "kv_cache": {"enabled": True, "targets": ["llama.cpp", "bitnet.cpp", "pytorch-cuda"]},
    }


@router.post("/api/playground/load")
async def playground_load(request: CompiledLoadRequest):
    artifacts = _compiled_artifacts()
    selected = next((item for item in artifacts if item["name"] == request.artifact), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="Artefato compilado não encontrado.")
    try:
        if request.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA não está disponível neste computador.")
            return await asyncio.to_thread(gpu_inference_service.preload)
        if selected["target"] == "retbit-cpp":
            return await asyncio.to_thread(
                compiled_service.preload, Path(str(selected["path"])), request.threads
            )
        service = native_service_for_target(str(selected["target"]))
        if service is None:
            raise RuntimeError(f"Não existe runtime para o alvo {selected['target']}.")
        return await asyncio.to_thread(
            service.preload, Path(str(selected["path"])), request.threads
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/playground/chat")
async def playground_chat(request: CompiledChatRequest):
    artifacts = _compiled_artifacts()
    if not artifacts:
        raise HTTPException(status_code=409, detail="Compile um modelo de fine-tuning antes de usar o Playground.")
    selected = next((item for item in artifacts if item["name"] == request.artifact), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="Artefato compilado não encontrado.")
    messages = [message.model_dump() for message in request.messages]
    try:
        sampling = _resolve_request_sampling(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def stream():
        try:
            if request.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA não está disponível neste computador.")
                began = time.perf_counter()
                output = ""
                for delta in gpu_inference_service.generate(
                    messages, request.max_new_tokens, sampling.temperature, sampling.top_p,
                    sampling.top_k, sampling.repetition_penalty, request.seed,
                ):
                    output += delta
                    yield json.dumps({"type": "token", "text": delta}, ensure_ascii=False) + "\n"
                elapsed = max(time.perf_counter() - began, 0.001)
                count = gpu_inference_service.last_generated_token_count
                yield json.dumps({
                    "type": "done", "token_count": count, "elapsed_seconds": round(elapsed, 3),
                    "tokens_per_second": round(count / elapsed, 2), "runtime": "retbit-cuda-i2-graph",
                    "device": "cuda",
                }, ensure_ascii=False) + "\n"
            else:
                runtime = (
                    compiled_service
                    if selected["target"] == "retbit-cpp"
                    else native_service_for_target(str(selected["target"]))
                )
                if runtime is None:
                    raise RuntimeError(f"Não existe runtime para o alvo {selected['target']}.")
                for event in runtime.stream(
                    Path(str(selected["path"])), messages, request.max_new_tokens,
                    sampling.temperature, sampling.top_p, sampling.top_k,
                    sampling.repetition_penalty, request.seed if request.seed is not None else 42,
                    request.threads,
                ):
                    yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson; charset=utf-8")
