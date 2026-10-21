import json
import random
import threading
from typing import Generator

import torch
from safetensors.torch import load_file

from src.architecture import architecture_from_checkpoint, is_attention, is_recurrent
from src.finetuning.config import FineTuningSettings
from src.finetuning.lora import attach_lora, load_adapter, parse_target_modules
from src.generation import advance_generation, prefill_generation
from src.finetuning.chat_format import format_generation_prompt
from src.model import CustomLLM
from src.tokenizer import decode_continuation, tokenizer, validate_tokenizer_metadata


class _CudaDecodeGraph:
    """Captured single-token decode graph with persistent RetNet state buffers."""

    def __init__(self, model: CustomLLM):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.token = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.states = [
            torch.zeros(
                (1, layer.retention.num_heads, layer.retention.head_dim, layer.retention.head_dim),
                dtype=dtype,
                device=device,
            )
            for layer in model.layers
        ]

        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                model.recurrent(self.token, self.states)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        for state in self.states:
            state.zero_()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.logits, _ = model.recurrent(self.token, self.states)

    def load_states(self, source: list[torch.Tensor]) -> None:
        if len(source) != len(self.states):
            raise ValueError("RetNet state count does not match the CUDA graph")
        for destination, state in zip(self.states, source, strict=True):
            destination.copy_(state)

    def step(self, token: torch.Tensor) -> torch.Tensor:
        self.token.copy_(token)
        self.graph.replay()
        return self.logits


class _CudaSampler:
    def __init__(self, vocabulary_size: int, device: torch.device):
        self.workspace = torch.empty(vocabulary_size, dtype=torch.float32, device=device)
        self.repetition_mask = torch.zeros(vocabulary_size, dtype=torch.uint8, device=device)
        self.output = torch.empty((1, 1), dtype=torch.long, device=device)
        self.random = random.Random()

    def reset(self, seed: int | None) -> None:
        self.repetition_mask.zero_()
        self.random.seed(seed)

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> torch.Tensor:
        from src.retbit_cuda import retbit_cuda_runtime

        greedy = temperature <= 0
        return retbit_cuda_runtime.sample_topk(
            logits,
            self.workspace,
            self.repetition_mask,
            self.output,
            repetition_penalty,
            1.0 if greedy else temperature,
            1.0 if greedy else top_p,
            1 if greedy else top_k,
            0.0 if greedy else self.random.random(),
        )


class FineTuningInferenceService:
    """Read-only inference from the latest atomically published SFT checkpoint."""

    def __init__(self, settings: FineTuningSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self._model: CustomLLM | None = None
        self._model_mtime = 0.0
        self._device = torch.device("cpu")
        self._decode_graph: _CudaDecodeGraph | None = None
        self._cuda_sampler: _CudaSampler | None = None
        self._last_generated_token_count = 0

    def set_settings(self, settings: FineTuningSettings) -> None:
        with self._lock:
            if settings.base_path == self.settings.base_path:
                self.settings = settings
                return
            self.settings = settings
            self._model = None
            self._model_mtime = 0.0
            self._decode_graph = None
            self._cuda_sampler = None

    def _load_model(self) -> CustomLLM:
        path = self.settings.base_path / "model.safetensor"
        if not path.exists():
            raise RuntimeError("O modelo base_finetuning ainda não existe.")
        config_path = self.settings.base_path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        is_lora = config.get("fine_tuning_method") == "lora"
        weights_path = (
            self.settings.base_model_path
            if is_lora and self.settings.base_model_path.is_file()
            else path
        )
        tracked_paths = [path, config_path]
        if is_lora:
            tracked_paths.extend([self.settings.base_model_path, self.settings.adapter_path])
        mtime = max(item.stat().st_mtime for item in tracked_paths if item.is_file())
        if self._model is not None and self._model_mtime == mtime:
            return self._model

        device_name = (
            self.settings.chat_device
            if self.settings.chat_device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self._device = torch.device(device_name)
        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        weights = load_file(str(weights_path), device=str(self._device))
        expected_vocab = int(weights["embeddings.weight"].shape[0])
        validate_tokenizer_metadata(config, expected_vocab)
        architecture = architecture_from_checkpoint(config, weights)
        model = CustomLLM(
            vocab_size=expected_vocab,
            d_model=int(config["model_d_model"]),
            num_layers=int(config["model_layers"]),
            architecture=architecture,
            num_heads=int(config["num_heads"]) if config.get("num_heads") else None,
            head_dim=int(config["head_dim"]) if config.get("head_dim") else None,
            num_kv_heads=int(config.get("num_kv_heads", 0) or 0),
            rope_theta=float(config.get("rope_theta", 10000.0)),
            max_context=int(config.get("context_window", self.settings.context_window)),
            ffn_hidden_size=int(config.get("ffn_hidden_size", 0) or 0),
            ffn_multiplier=float(config.get("ffn_multiplier", 4.0)),
            ffn_multiple_of=int(config.get("ffn_multiple_of", 64)),
            norm_eps=float(config.get("norm_eps", 1e-6)),
            init_std=float(config.get("init_std", 0.02)),
        ).to(device=self._device, dtype=dtype)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if set(missing) - {"lm_head.weight"} or unexpected:
            raise RuntimeError("O checkpoint de fine-tuning não corresponde ao modelo atual.")
        if is_lora:
            attach_lora(
                model,
                rank=int(config.get("fine_tuning_lora_rank", self.settings.lora_rank)),
                alpha=float(config.get("fine_tuning_lora_alpha", self.settings.lora_alpha)),
                dropout=0.0,
                target_modules=parse_target_modules(
                    str(config.get("fine_tuning_lora_target_modules", self.settings.lora_target_modules))
                ),
            )
            load_adapter(model, self.settings.adapter_path)
        model.eval()
        model.prepare_for_inference()
        # O CUDA graph de decode é exclusivo das arquiteturas com estado recorrente
        # fixo (RetNet/RetBit). Transformer/BitNet usam KV cache crescente via recurrent().
        self._decode_graph = (
            _CudaDecodeGraph(model)
            if self._device.type == "cuda" and is_recurrent(model.architecture)
            else None
        )
        from src.retbit_cuda import retbit_cuda_runtime

        self._cuda_sampler = (
            _CudaSampler(expected_vocab, self._device)
            if self._device.type == "cuda" and retbit_cuda_runtime.available
            else None
        )
        self._model = model
        self._model_mtime = mtime
        return model

    def preload(self) -> dict[str, str | int | float]:
        with self._lock:
            model = self._load_model()
            parameters = sum(parameter.numel() for parameter in model.parameters())
            return {
                "loaded": True,
                "device": str(self._device),
                "parameters": parameters,
                "architecture": self._model.architecture.value,
                "checkpoint_mtime": self._model_mtime,
                "runtime": (
                    "retbit-cuda-i2-graph"
                    if self._device.type == "cuda" and self._cuda_sampler is not None and is_recurrent(self._model.architecture)
                    else "pytorch-kv-cache" if is_attention(self._model.architecture)
                    else "pytorch-recurrent"
                ),
                "kv_cache": {
                    "enabled": is_attention(self._model.architecture),
                    "implementation": "prefill + incremental K/V" if is_attention(self._model.architecture) else "estado RetNet",
                },
            }

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def last_generated_token_count(self) -> int:
        return self._last_generated_token_count

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        return format_generation_prompt(messages, self.settings.system_prompt)

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        seed: int | None,
    ) -> Generator[str, None, None]:
        with self._lock:
            model = self._load_model()
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            prompt = self._format_prompt(messages)
            full_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self._device)
            generated: list[int] = []
            self._last_generated_token_count = 0
            decoded_so_far = ""
            eos_id = tokenizer.eos_token_id
            context_window = int(model.max_context)
            limit = max(1, min(max_new_tokens, context_window))

            with torch.inference_mode():
                visible_ids, logits, states = prefill_generation(
                    model, full_prompt_ids, context_window
                )
                prompt_token_ids = full_prompt_ids[0].tolist()
                decode_graph = self._decode_graph
                cuda_sampler = self._cuda_sampler
                if decode_graph is not None:
                    decode_graph.load_states(states)
                if cuda_sampler is not None:
                    cuda_sampler.reset(seed)
                for index in range(limit):
                    use_cuda_sampler = cuda_sampler is not None and (
                        temperature <= 0 or 0 < top_k <= 256
                    )
                    if use_cuda_sampler:
                        next_token = cuda_sampler.sample(
                            logits,
                            temperature,
                            top_p,
                            top_k,
                            repetition_penalty,
                        )
                    else:
                        sampling_logits = logits.float()
                        if repetition_penalty != 1.0 and generated:
                            repeated = torch.tensor(
                                list(set(generated)), dtype=torch.long, device=self._device
                            )
                            values = sampling_logits[0, repeated]
                            sampling_logits[0, repeated] = torch.where(
                                values < 0,
                                values * repetition_penalty,
                                values / repetition_penalty,
                            )

                        if temperature <= 0:
                            next_token = torch.argmax(sampling_logits, dim=-1, keepdim=True)
                        else:
                            sampling_logits = sampling_logits / max(temperature, 1e-5)
                            candidate_count = min(top_k, sampling_logits.size(-1)) if top_k > 0 else 0
                            if candidate_count:
                                candidate_logits, candidate_ids = torch.topk(
                                    sampling_logits, candidate_count, dim=-1
                                )
                            else:
                                candidate_logits, candidate_ids = torch.sort(
                                    sampling_logits, descending=True
                                )
                            if top_p < 1.0:
                                cumulative = torch.cumsum(torch.softmax(candidate_logits, dim=-1), dim=-1)
                                remove = cumulative > top_p
                                remove[:, 1:] = remove[:, :-1].clone()
                                remove[:, 0] = False
                                candidate_logits = candidate_logits.masked_fill(remove, float("-inf"))
                            probabilities = torch.softmax(candidate_logits, dim=-1)
                            selected = torch.multinomial(probabilities, num_samples=1)
                            next_token = torch.gather(candidate_ids, 1, selected)

                    token_id = int(next_token.item())
                    if eos_id is not None and token_id == eos_id:
                        break
                    generated.append(token_id)
                    decoded = decode_continuation(prompt_token_ids, generated)
                    delta = decoded[len(decoded_so_far) :] if decoded.startswith(decoded_so_far) else decoded
                    decoded_so_far = decoded
                    if delta:
                        yield delta
                    if index + 1 < limit:
                        if visible_ids.size(1) >= context_window:
                            visible_ids = torch.cat([visible_ids, next_token], dim=-1)
                            visible_ids, logits, states = prefill_generation(
                                model, visible_ids, context_window
                            )
                            if decode_graph is not None:
                                decode_graph.load_states(states)
                        elif decode_graph is not None:
                            visible_ids = torch.cat([visible_ids, next_token], dim=-1)
                            logits = decode_graph.step(next_token)
                        else:
                            visible_ids, logits, states, _ = advance_generation(
                                model,
                                visible_ids,
                                next_token,
                                states,
                                context_window,
                            )
                self._last_generated_token_count = len(generated)
