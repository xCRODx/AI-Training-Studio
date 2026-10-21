import asyncio
import json
import logging
import os
import shutil
from typing import Optional, AsyncGenerator, Literal

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import load_file

from src.config import settings
from src.model import CustomLLM
from src.tokenizer import decode_continuation, tokenizer
from src.train import train_loop, CHECKPOINT_DIR, CHECKPOINT_CONFIG
from src.base_inference import BaseCheckpointInference, snapshot_live_model
from src.generation import advance_generation, prefill_generation
from src.sampling import resolve_sampling
from src.training_lock import training_lease
from src.lifecycle import shutdown_requested
from src.finetuning.manager import finetuning_manager
from src.finetuning.router import router as finetuning_router
import src.train as train_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração do FastAPI
app = FastAPI(title="Custom LLM Trainer & Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Arquivos estáticos (garante que as pastas existam)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("docs", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/docs", StaticFiles(directory="docs"), name="docs")
app.include_router(finetuning_router)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

# Estado Global
class AppState:
    is_training: dict = {
        "active": False,
        "pause_requested": False,
        "force_save": False,
        "checkpoint_saved": True,
        "shutdown_requested": False,
    }
    training_task: Optional[asyncio.Task] = None
    metrics_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    inference_model: Optional[CustomLLM] = None

state = AppState()
base_inference = BaseCheckpointInference(CHECKPOINT_DIR, settings.context_window)
finetuning_manager.set_resource_guard(lambda: not state.is_training["active"])
finetuning_manager.set_training_lease(training_lease)


async def _run_pretraining_with_lease() -> None:
    try:
        await train_loop(state.metrics_queue, state.is_training)
    finally:
        training_lease.release()


@app.on_event("shutdown")
async def release_training_lease_on_shutdown() -> None:
    shutdown_requested.set()
    if state.training_task is not None and not state.training_task.done():
        if state.is_training.get("active", False):
            state.is_training["checkpoint_saved"] = False
            state.is_training["pause_requested"] = True
        state.is_training["shutdown_requested"] = True
        deadline = asyncio.get_running_loop().time() + 120
        while (
            not state.is_training.get("checkpoint_saved", False)
            and not state.training_task.done()
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.1)
    training_lease.release()

# Pydantic Models
class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    sampling_mode: Literal["auto", "greedy", "sample"] = "auto"
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    device: str = "cuda"


# Rotas de Controle de Treinamento
@app.post("/api/train/start")
async def start_training():
    if state.is_training["active"]:
        raise HTTPException(status_code=400, detail="O treinamento já está em andamento.")
    if finetuning_manager.is_active():
        raise HTTPException(status_code=409, detail="Pause ou pare o fine-tuning antes de iniciar o pré-treino.")
    
    task_alive = state.training_task is not None and not state.training_task.done()
    if not task_alive:
        try:
            training_lease.acquire("pré-treino")
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    state.is_training["active"] = True
    state.is_training["pause_requested"] = False
    state.is_training["force_save"] = False
    state.is_training["checkpoint_saved"] = False
    state.is_training["shutdown_requested"] = False

    if task_alive:
        return {"status": "success", "state": "running", "message": "Treinamento retomado."}

    while not state.metrics_queue.empty():
        state.metrics_queue.get_nowait()
    state.training_task = asyncio.create_task(_run_pretraining_with_lease())
    return {"status": "success", "state": "running", "message": "Treinamento iniciado."}


@app.post("/api/train/resume")
async def resume_training():
    if state.is_training["active"]:
        raise HTTPException(status_code=400, detail="O treinamento já está em andamento.")
    if state.training_task is None or state.training_task.done():
        raise HTTPException(status_code=409, detail="Não existe um treino pausado para retomar.")
    if finetuning_manager.is_active():
        raise HTTPException(status_code=409, detail="Pause ou pare o fine-tuning antes de retomar o pré-treino.")
    state.is_training["active"] = True
    state.is_training["pause_requested"] = False
    return {"status": "success", "state": "running", "message": "Treinamento retomado."}


@app.get("/api/train/status")
async def training_status():
    task_alive = state.training_task is not None and not state.training_task.done()
    current = (
        "pausing"
        if state.is_training.get("pause_requested", False)
        else "running"
        if state.is_training["active"]
        else "paused"
        if task_alive
        else "idle"
    )
    from src.architecture import (
        architecture_from_config,
        architecture_from_state_dict,
        architecture_label,
        resolve_architecture,
    )

    checkpoint_arch = architecture_from_config(CHECKPOINT_CONFIG)
    if checkpoint_arch is None and os.path.exists(os.path.join(CHECKPOINT_DIR, "model.safetensor")):
        try:
            with safe_open(os.path.join(CHECKPOINT_DIR, "model.safetensor"), framework="pt", device="cpu") as handle:
                checkpoint_arch = architecture_from_state_dict(dict.fromkeys(handle.keys()))
        except Exception:
            checkpoint_arch = None
    architecture = checkpoint_arch or resolve_architecture()
    from src.architecture import is_attention

    return {
        "state": current,
        "active": state.is_training["active"],
        "pause_requested": state.is_training.get("pause_requested", False),
        "checkpoint_saved": state.is_training.get("checkpoint_saved", False),
        "initialized": task_alive or train_module.LIVE_MODEL is not None,
        "checkpoint_available": os.path.isfile(os.path.join(CHECKPOINT_DIR, "model.safetensor")),
        "architecture": architecture.value,
        "architecture_label": architecture_label(architecture),
        "use_bitnet": settings.use_bitnet,
        "use_retnet": settings.use_retnet,
        "source": "checkpoint" if checkpoint_arch is not None else "env",
        "kv_cache": {
            "enabled": is_attention(architecture),
            "training": False,
            "inference": is_attention(architecture),
            "implementation": "prefill + incremental K/V" if is_attention(architecture) else "estado RetNet",
        },
    }

@app.get("/api/train/metrics/history")
async def training_metrics_history():
    """Lê o training_metrics.csv e devolve as séries de step/loss para o gráfico
    de média móvel (EMA) da loss, mostrando a tendência real escondida pelo ruído."""
    import csv as csv_module
    csv_path = os.path.join(CHECKPOINT_DIR, "training_metrics.csv")
    steps: list[int] = []
    losses: list[float] = []
    perplexity: list[float] = []
    val_loss: list[float] = []
    top1: list[float] = []
    entropy: list[float] = []
    if not os.path.exists(csv_path):
        return {
            "steps": steps, "losses": losses,
            "perplexity": perplexity, "val_loss": val_loss,
            "top1": top1, "entropy": entropy,
        }
    try:
        def _num(value):
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                step = row.get("Step")
                loss = row.get("Loss")
                if step is None or loss is None or loss == "":
                    continue
                try:
                    steps.append(int(float(step)))
                    losses.append(float(loss))
                    perplexity.append(_num(row.get("Perplexity")))
                    val_loss.append(_num(row.get("Val_Loss")))
                    top1.append(_num(row.get("Acc_Top1")))
                    entropy.append(_num(row.get("Logits_Entropy")))
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        logger.error(f"Erro ao ler histórico de métricas: {e}")
        return {"steps": [], "losses": [], "perplexity": [], "val_loss": [], "top1": [], "entropy": []}
    return {
        "steps": steps, "losses": losses,
        "perplexity": perplexity, "val_loss": val_loss,
        "top1": top1, "entropy": entropy,
    }


@app.post("/api/train/stop")
async def stop_training():
    if not state.is_training["active"]:
        raise HTTPException(status_code=400, detail="O treinamento não está rodando.")
    
    # Mantém o loop ativo até a próxima fronteira do optimizer. Isso garante que
    # nenhum gradiente parcial da acumulação seja perdido ao fechar o Python.
    state.is_training["checkpoint_saved"] = False
    state.is_training["pause_requested"] = True
    deadline = asyncio.get_running_loop().time() + 300
    while (
        not state.is_training.get("checkpoint_saved", False)
        and state.training_task is not None
        and not state.training_task.done()
        and asyncio.get_running_loop().time() < deadline
    ):
        await asyncio.sleep(0.1)
    if not state.is_training.get("checkpoint_saved", False):
        raise HTTPException(
            status_code=504,
            detail="A pausa segura não terminou dentro do prazo; não finalize o Python.",
        )
    return {
        "status": "success",
        "state": "paused",
        "checkpoint_saved": True,
        "message": "Treinamento pausado em fronteira segura e checkpoint confirmado.",
    }


# Gerenciamento de Checkpoints
@app.post("/api/checkpoint/create")
async def create_checkpoint():
    if state.is_training["active"]:
        # Se o treinamento estiver rodando, agenda o salvamento
        state.is_training["force_save"] = True
        return {"status": "success", "message": "Salvamento forçado agendado para o final do batch atual."}

    # Se o treinamento estiver pausado, tenta copiar do diretório base
    if not os.path.exists(CHECKPOINT_CONFIG):
        raise HTTPException(status_code=400, detail="Treine o modelo antes de tentar criar um checkpoint.")
    
    try:
        with open(CHECKPOINT_CONFIG, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            tokens_consumed = config_data.get("tokens_consumed", 0)
            
        new_checkpoint_dir = await asyncio.to_thread(
            train_module.create_persistent_snapshot, tokens_consumed
        )
        return {"status": "success", "message": "Checkpoint criado com sucesso.", "path": new_checkpoint_dir}
    except Exception as e:
        logger.error(f"Erro ao criar checkpoint offline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao criar checkpoint: {str(e)}")


# Rota de Inferência
@app.post("/api/infer")
async def infer(request: InferenceRequest):
    device_name = request.device if torch.cuda.is_available() and request.device == "cuda" else "cpu"

    try:
        # A live model is never switched to eval() in place. A stable snapshot
        # preserves the exact architecture/weights at request start while the
        # optimizer remains free to continue training its own model instance.
        model = (
            snapshot_live_model(train_module.LIVE_MODEL, device_name)
            if train_module.LIVE_MODEL is not None
            else await asyncio.to_thread(base_inference.load, device_name)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    
    actual_device = next(model.parameters()).device
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt", add_special_tokens=False).to(actual_device)
    try:
        sampling = resolve_sampling(
            request.sampling_mode,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            auto_temperature=settings.inference_auto_temperature,
            auto_top_p=settings.inference_auto_top_p,
            auto_top_k=settings.inference_auto_top_k,
            auto_repetition_penalty=settings.inference_auto_repetition_penalty,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        context_window = int(model.max_context)
        visible_ids, next_token_logits, states = prefill_generation(model, input_ids, context_window)
        prompt_token_ids = input_ids[0].tolist()
        generated: list[int] = []
        decoded_so_far = ""

        for index in range(max(1, min(request.max_new_tokens, context_window))):
            with torch.no_grad():
                sampling_logits = next_token_logits.float()

                if sampling.repetition_penalty != 1.0 and generated:
                    repeated = torch.tensor(
                        list(set(generated)), dtype=torch.long, device=actual_device
                    )
                    values = sampling_logits[0, repeated]
                    sampling_logits[0, repeated] = torch.where(
                        values < 0,
                        values * sampling.repetition_penalty,
                        values / sampling.repetition_penalty,
                    )

                if sampling.mode == "greedy":
                    next_token = torch.argmax(sampling_logits, dim=-1, keepdim=True)
                else:
                    sampling_logits = sampling_logits / sampling.temperature

                    if sampling.top_k > 0:
                        keep = min(sampling.top_k, sampling_logits.size(-1))
                        threshold = torch.topk(sampling_logits, keep, dim=-1).values[:, -1:]
                        sampling_logits = sampling_logits.masked_fill(
                            sampling_logits < threshold, float("-inf")
                        )

                    if sampling.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(sampling_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        remove = cumulative_probs > sampling.top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        mask = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
                        sampling_logits = sampling_logits.masked_fill(mask, float("-inf"))

                    probs = torch.softmax(sampling_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                token_id = int(next_token.item())
                if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                    break
                generated.append(token_id)
                decoded = decode_continuation(prompt_token_ids, generated)
                delta = decoded[len(decoded_so_far) :] if decoded.startswith(decoded_so_far) else decoded
                decoded_so_far = decoded

                if index + 1 < max(1, min(request.max_new_tokens, context_window)):
                    visible_ids, next_token_logits, states, _ = advance_generation(
                        model, visible_ids, next_token, states, context_window
                    )

            if delta:
                yield delta
            await asyncio.sleep(0)
                
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# WebSocket de Telemetria
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    logger.info("Cliente WebSocket conectado.")
    
    try:
        while not shutdown_requested.is_set():
            # Aguarda a próxima métrica
            try:
                metric = await asyncio.wait_for(state.metrics_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await websocket.send_json(metric)
    except WebSocketDisconnect:
        logger.info("Cliente WebSocket desconectado.")
    except Exception as e:
        logger.error(f"Erro no WebSocket: {e}")
    finally:
        # Garante que não trava se fechar inesperadamente
        pass
