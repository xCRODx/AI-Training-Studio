from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.architecture import Architecture, is_attention, is_recurrent, resolve_architecture
from src.config import settings


def ffn_hidden_dim(
    d_model: int,
    hidden_size: int = 0,
    multiplier: float = 4.0,
    multiple_of: int = 64,
) -> int:
    """Resolve FFN width from an explicit size or a configurable ratio."""
    if hidden_size > 0:
        return int(hidden_size)
    if multiplier <= 0 or multiple_of <= 0:
        raise ValueError("FFN multiplier e multiple_of devem ser positivos.")
    raw_hidden = math.ceil(d_model * multiplier)
    return ((raw_hidden + multiple_of - 1) // multiple_of) * multiple_of


class BitLinear(nn.Module):
    """
    Camada Linear Quantizada (BitLinear) 1.58-bit.
    Simula a lógica da BitNet onde os pesos são quantizados para {-1, 0, 1}.

    Quando ``quantized=False`` a camada se comporta como um ``nn.Linear`` comum
    (usado pelas arquiteturas RetNet float e Transformer padrão), mantendo a
    mesma assinatura de state_dict.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, quantized: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantized = quantized
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        # Built explicitly by the inference service and excluded from state_dict.
        self._inference_weight: torch.Tensor | None = None
        self._inference_packed_weight: torch.Tensor | None = None
        self._inference_scale = 1.0

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        """Materialize the effective ternary matrix once for low-latency decode."""
        if not self.quantized:
            return
        scale = self.weight.abs().mean().clamp(min=1e-5)
        quantized = torch.round(self.weight / scale).clamp(-1, 1)
        self._inference_weight = (quantized * scale).contiguous()
        self._inference_scale = float(scale.float().item())
        if self.weight.is_cuda and self.in_features % 4 == 0:
            codes = (quantized.reshape(-1).to(torch.int16) + 1).to(torch.uint8)
            self._inference_packed_weight = (
                codes[0::4]
                | (codes[1::4] << 2)
                | (codes[2::4] << 4)
                | (codes[3::4] << 6)
            ).contiguous()

    def clear_inference_cache(self) -> None:
        self._inference_weight = None
        self._inference_packed_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            return F.linear(x, self.weight.to(dtype=x.dtype, device=x.device), self.bias)

        if not self.training and self._inference_weight is not None:
            if (
                x.is_cuda
                and x.dtype == torch.bfloat16
                and x.numel() // self.in_features == 1
                and self._inference_packed_weight is not None
            ):
                from src.retbit_cuda import retbit_cuda_runtime

                if retbit_cuda_runtime.available:
                    return retbit_cuda_runtime.linear(
                        x,
                        self._inference_packed_weight,
                        self._inference_scale,
                        self.out_features,
                    )
            return F.linear(x, self._inference_weight, self.bias)

        # 1. Armazena o dtype de entrada (ex: torch.bfloat16)
        target_dtype = x.dtype

        # 2. Garante que os pesos latentes estão no mesmo device e tipo para a operação
        w = self.weight.to(device=x.device, dtype=target_dtype)

        # 3. CRÍTICO: REMOVA o .detach(). A escala PRECISA propagar gradiente de magnitude!
        scale = w.abs().mean().clamp(min=1e-5)

        weight_scaled = w / scale
        weight_quantized = torch.round(weight_scaled).clamp(-1, 1)

        # 4. STE (Straight-Through Estimator) mantendo consistência de dtypes
        weight_ste = (weight_quantized - weight_scaled).detach() + weight_scaled

        # 5. Executa a operação linear e re-escala
        output = F.linear(x, weight_ste) * scale

        if self.bias is not None:
            output += self.bias.to(dtype=target_dtype)

        return output


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Positional Embedding) no estilo llama.cpp (pares intercalados).
    As frequências são pré-computadas até ``max_context``.
    """
    def __init__(self, dim: int, max_context: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_context, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("freqs_cos", freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.freqs_cos[:, :, positions]
        sin = self.freqs_sin[:, :, positions]
        pairs = x.view(*x.shape[:-1], -1, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack(
            [x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1
        )
        return rotated.view(*x.shape)


class MultiHeadAttention(nn.Module):
    """
    Atenção padrão com RoPE, Grouped-Query Attention opcional e cache de K/V
    incremental. Usada pelas arquiteturas ``transformer`` e ``bitnet``.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_context: int,
        quantized: bool,
        rope_theta: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_context = int(max_context)
        self.q_proj = BitLinear(d_model, num_heads * head_dim, quantized=quantized)
        self.k_proj = BitLinear(d_model, num_kv_heads * head_dim, quantized=quantized)
        self.v_proj = BitLinear(d_model, num_kv_heads * head_dim, quantized=quantized)
        self.out_proj = BitLinear(num_heads * head_dim, d_model, quantized=quantized)
        self.rope = RotaryPositionalEmbedding(head_dim, max_context, rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = cache[0].size(2) if cache is not None else 0
        if past_len + seq_len > self.max_context:
            raise ValueError(
                "Sequencia excede o contexto posicional do modelo: "
                f"passado={past_len}, atual={seq_len}, limite={self.max_context}. "
                "Use no maximo o context_window registrado no checkpoint."
            )
        q_positions = torch.arange(seq_len, device=x.device) + past_len

        q = self.rope.apply(q, q_positions)
        k = self.rope.apply(k, q_positions)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        # Guarda o cache ANTES da expansão GQA (só KV heads originais).
        k_cache, v_cache = k, v

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        total = k.size(2)
        # Mask causal por posições absolutas: query em p atende a k em 0..p.
        # No decode incremental (seq_len == 1) o cache já contém só passado,
        # portanto a linha única atende a todas as posições cacheadas.
        causal = (q_positions.view(seq_len, 1) >= torch.arange(total, device=x.device).view(1, total))
        attn_weights = attn_weights.masked_fill(~causal, float("-inf"))

        attn = torch.softmax(attn_weights, dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        return out, (k_cache.contiguous(), v_cache.contiguous())


class MultiScaleRetention(nn.Module):
    """
    Mecanismo de Retenção (Multi-Scale Retention - RetNet).
    Implementa a representação paralela substituindo o Softmax por decaimento causal exponencial.
    """
    def __init__(
        self, d_model: int, num_heads: int, quantized: bool = True, norm_eps: float = 1e-6
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projeções usando BitLinear (ternário preservado) ou Linear float.
        self.q_proj = BitLinear(d_model, d_model, quantized=quantized)
        self.k_proj = BitLinear(d_model, d_model, quantized=quantized)
        self.v_proj = BitLinear(d_model, d_model, quantized=quantized)
        # Gate swish (também ternário) — aumenta a não-linearidade, conforme o paper.
        self.g_proj = BitLinear(d_model, d_model, quantized=quantized)
        self.out_proj = BitLinear(d_model, d_model, quantized=quantized)

        # GroupNorm por head: equilibra as variâncias dos heads multiescala (γ diferentes).
        # Sem ele, os heads de memória longa (γ≈1) acumulam somas gigantes e afogam os demais,
        # impedindo o modelo de lembrar contexto de 10–20 tokens atrás.
        self.gn = nn.GroupNorm(num_heads, d_model, eps=norm_eps, affine=True)

        # Valores de decaimento gamma para cada head (ex: 1-1/32, 1-1/64...)
        gammas = [1 - 2**(-5 - i) for i in range(num_heads)]
        self.register_buffer("gammas", torch.tensor(gammas))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape

        q = self.q_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        g = self.g_proj(x)  # (b, seq, d_model) para o gate swish

        # Q * K^T (batch, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Decaimento causal exponencial D
        n = torch.arange(seq_len, device=x.device).unsqueeze(1)
        m = torch.arange(seq_len, device=x.device).unsqueeze(0)
        distance = n - m

        causal_mask = (distance >= 0).float()

        # Aplica o decaimento gamma correspondente para cada head
        gammas = self.gammas.view(-1, 1, 1)
        decay_mask = (gammas ** distance.clamp(min=0)) * causal_mask

        # Substitui a operação padrão de Softmax pelo decaimento (D)
        retention = attn_weights * decay_mask.unsqueeze(0)

        # Sincroniza os tipos de dados antes da multiplicação matricial
        retention = retention.to(v.dtype)
        out = torch.matmul(retention, v)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)

        # GroupNorm por head + gate swish (MSR do paper)
        out = self.gn(out.reshape(b * seq_len, self.d_model)).view(b, seq_len, self.d_model)
        out = out * torch.nn.functional.silu(g)

        return self.out_proj(out)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the prompt in parallel and return its final recurrent state."""
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        g = self.g_proj(x)

        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seq_len, device=x.device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        causal = distance >= 0
        decay = self.gammas.view(-1, 1, 1) ** distance.clamp(min=0)
        retention = attention * (decay * causal).unsqueeze(0).to(attention.dtype)
        output = torch.matmul(retention, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # GroupNorm por head + gate swish (igual ao forward, para consistência treino/inferência)
        output = self.gn(output.reshape(batch * seq_len, self.d_model)).view(batch, seq_len, self.d_model)
        output = output * torch.nn.functional.silu(g)

        reverse_distance = torch.arange(seq_len - 1, -1, -1, device=x.device)
        final_decay = self.gammas.view(-1, 1) ** reverse_distance.view(1, -1)
        state = torch.einsum("bhsd,bhse,hs->bhde", k, v, final_decay.to(k.dtype))
        return self.out_proj(output), state.contiguous()

    def recurrent(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one token with the fixed-size RetNet state."""
        batch, seq_len, _ = x.shape
        if seq_len != 1:
            raise ValueError("RetNet recurrent decode expects exactly one token")
        q = self.q_proj(x).view(batch, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, self.num_heads, self.head_dim)
        g = self.g_proj(x)  # (b, 1, d_model)
        state.mul_(self.gammas.view(1, -1, 1, 1).to(state.dtype))
        state.add_(k.unsqueeze(-1) * v.unsqueeze(-2))
        output = torch.matmul(q.unsqueeze(-2), state).squeeze(-2) / math.sqrt(self.head_dim)
        output = output.reshape(batch, self.d_model)
        # GroupNorm por head + gate swish (consistente com forward/prefill)
        output = self.gn(output)
        output = output * torch.nn.functional.silu(g.reshape(batch, self.d_model))
        output = output.reshape(batch, 1, self.d_model)
        return self.out_proj(output), state


class FeedForward(nn.Module):
    """Rede Feed-Forward densa padrão (SiLU) — RetNet/RetBit."""
    def __init__(self, d_model: int, hidden_dim: int, quantized: bool = True):
        super().__init__()
        self.fc1 = BitLinear(d_model, hidden_dim, quantized=quantized)
        self.act = nn.SiLU()
        self.fc2 = BitLinear(hidden_dim, d_model, quantized=quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GatedFeedForward(nn.Module):
    """Rede Feed-Forward com gate SiLU (SwiGLU) — Transformer/BitNet."""
    def __init__(self, d_model: int, hidden_dim: int, quantized: bool = True):
        super().__init__()
        self.gate_proj = BitLinear(d_model, hidden_dim, quantized=quantized)
        self.up_proj = BitLinear(d_model, hidden_dim, quantized=quantized)
        self.down_proj = BitLinear(hidden_dim, d_model, quantized=quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    """Normalização da raiz quadrada média."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RetNetBlock(nn.Module):
    """Bloco principal Denso combinando RetNet e FFN."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        quantized: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.retention = MultiScaleRetention(
            d_model, num_heads, quantized=quantized, norm_eps=norm_eps
        )

        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = FeedForward(d_model, ffn_dim, quantized=quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.retention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        retained, state = self.retention.prefill(self.norm1(x))
        x = x + retained
        x = x + self.ffn(self.norm2(x))
        return x, state

    def recurrent(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        retained, state = self.retention.recurrent(self.norm1(x), state)
        x = x + retained
        x = x + self.ffn(self.norm2(x))
        return x, state


class TransformerBlock(nn.Module):
    """Bloco Transformer padrão (attention + SwiGLU FFN) com KV cache."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_context: int,
        ffn_dim: int,
        quantized: bool,
        rope_theta: float,
        norm_eps: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attention = MultiHeadAttention(
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            max_context,
            quantized,
            rope_theta,
        )

        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = GatedFeedForward(d_model, ffn_dim, quantized=quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(self.norm1(x))
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return x

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, cache = self.attention(self.norm1(x))
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return x, cache

    def recurrent(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, cache = self.attention(self.norm1(x), cache)
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return x, cache


def _default_kv_heads(num_heads: int) -> int:
    """Escolhe um divisor 'redondo' dos Q heads para GQA (1 se nenhum servir)."""
    for candidate in (num_heads // 8, num_heads // 4, num_heads // 2):
        if candidate >= 1 and num_heads % candidate == 0:
            return candidate
    return 1


class CustomLLM(nn.Module):
    """
    O Modelo Principal (CustomLLM).

    Suporta quatro arquiteturas selecionadas pelas flags ``USE_BITNET``/
    ``USE_RETNET`` (ou pelo parâmetro ``architecture``):

    - ``transformer``: attention padrão (RoPE + GQA + SwiGLU) com KV cache.
    - ``bitnet``:      o mesmo transformer, porém com projeções BitLinear ternárias.
    - ``retnet``:      RetNet com projeções float.
    - ``retbit``:      RetNet + BitLinear (híbrido legado, preserva o comportamento atual).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int | None = None,
        num_layers: int | None = None,
        architecture: str | Architecture | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        rope_theta: float | None = None,
        max_context: int | None = None,
        ffn_hidden_size: int | None = None,
        ffn_multiplier: float | None = None,
        ffn_multiple_of: int | None = None,
        norm_eps: float | None = None,
        init_std: float | None = None,
    ):
        super().__init__()
        # Optional architecture values let fine-tuning/distributed backends build
        # the exact checkpoint shape. Existing pre-training calls remain unchanged.
        self.d_model = d_model if d_model is not None else settings.model_d_model
        self.num_layers = num_layers if num_layers is not None else settings.model_layers
        self.architecture = Architecture(architecture or resolve_architecture())

        configured_heads = num_heads if num_heads is not None else settings.model_num_heads
        configured_head_dim = head_dim if head_dim is not None else settings.model_head_dim
        if configured_heads > 0:
            if self.d_model % configured_heads != 0:
                raise ValueError("MODEL_D_MODEL deve ser divisível por MODEL_NUM_HEADS.")
            self.num_heads = int(configured_heads)
            self.head_dim = self.d_model // self.num_heads
            if head_dim is not None and head_dim > 0 and head_dim != self.head_dim:
                raise ValueError("num_heads/head_dim do checkpoint são inconsistentes com d_model.")
        else:
            if configured_head_dim <= 0 or self.d_model % configured_head_dim != 0:
                raise ValueError("MODEL_HEAD_DIM deve ser positivo e dividir MODEL_D_MODEL.")
            self.head_dim = int(configured_head_dim)
            self.num_heads = self.d_model // self.head_dim
        candidate_kv = num_kv_heads if num_kv_heads is not None and num_kv_heads > 0 else settings.model_kv_heads
        if candidate_kv > 0 and self.num_heads % candidate_kv == 0:
            self.num_kv_heads = candidate_kv
        else:
            self.num_kv_heads = _default_kv_heads(self.num_heads)
        self.rope_theta = rope_theta if rope_theta is not None else settings.rope_theta
        self.max_context = max_context if max_context is not None else settings.context_window
        self.ffn_multiplier = (
            ffn_multiplier if ffn_multiplier is not None else settings.model_ffn_multiplier
        )
        self.ffn_multiple_of = (
            ffn_multiple_of if ffn_multiple_of is not None else settings.model_ffn_multiple_of
        )
        requested_ffn = (
            ffn_hidden_size if ffn_hidden_size is not None else settings.model_ffn_hidden_size
        )
        self.ffn_hidden_size = ffn_hidden_dim(
            self.d_model, requested_ffn, self.ffn_multiplier, self.ffn_multiple_of
        )
        self.norm_eps = norm_eps if norm_eps is not None else settings.model_norm_eps
        self.init_std = init_std if init_std is not None else settings.model_init_std

        quantized = self.architecture in (Architecture.BITNET, Architecture.RETBIT)

        self.embeddings = nn.Embedding(vocab_size, self.d_model)
        torch.nn.init.normal_(self.embeddings.weight, mean=0.0, std=self.init_std)

        if is_recurrent(self.architecture):
            self.layers = nn.ModuleList([
                RetNetBlock(
                    self.d_model,
                    self.num_heads,
                    self.ffn_hidden_size,
                    quantized=quantized,
                    norm_eps=self.norm_eps,
                )
                for _ in range(self.num_layers)
            ])
        elif is_attention(self.architecture):
            self.layers = nn.ModuleList([
                TransformerBlock(
                    self.d_model,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.max_context,
                    self.ffn_hidden_size,
                    quantized=quantized,
                    rope_theta=self.rope_theta,
                    norm_eps=self.norm_eps,
                )
                for _ in range(self.num_layers)
            ])
        else:  # pragma: no cover - guarded by the enum
            raise ValueError(f"Arquitetura desconhecida: {self.architecture}")

        self.norm_f = RMSNorm(self.d_model, eps=self.norm_eps)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)

        # Weight Tying: compartilhamento de pesos entre a entrada e saída (economiza ~7.6M parâmetros)
        self.lm_head.weight = self.embeddings.weight

        # Gradient Checkpointing (recomputa o forward no backward para economizar VRAM).
        # Só ativo no treino; prefill/recurrent (inferência) não passam por aqui.
        self.gradient_checkpointing = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.prepare_for_inference()

    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[AnyState]]:
        x = self.embeddings(input_ids)
        states: list[AnyState] = []
        for layer in self.layers:
            x, state = layer.prefill(x)
            states.append(state)
        logits = self.lm_head(self.norm_f(x[:, -1:, :]))[:, -1, :]
        return logits, states

    def recurrent(
        self,
        input_ids: torch.Tensor,
        states: list[AnyState],
    ) -> tuple[torch.Tensor, list[AnyState]]:
        if len(states) != len(self.layers):
            raise ValueError("State count does not match the model")
        x = self.embeddings(input_ids)
        for index, layer in enumerate(self.layers):
            x, states[index] = layer.recurrent(x, states[index])
        logits = self.lm_head(self.norm_f(x))[:, -1, :]
        return logits, states

    def architecture_info(self) -> dict:
        return {
            "architecture": self.architecture.value,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "ffn_hidden_size": self.ffn_hidden_size,
            "ffn_multiplier": self.ffn_multiplier,
            "ffn_multiple_of": self.ffn_multiple_of,
            "norm_eps": self.norm_eps,
            "init_std": self.init_std,
            "num_kv_heads": self.num_kv_heads,
            "rope_theta": self.rope_theta,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        }


AnyState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
