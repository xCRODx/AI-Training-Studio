from collections.abc import Sequence
from functools import lru_cache
import hashlib
import logging

from transformers import AutoTokenizer, PreTrainedTokenizer
import os
from src.config import settings

# Configuração de log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tokenizer() -> PreTrainedTokenizer:
    """
    Carrega o tokenizer especificado no arquivo .env.
    O download será feito e armazenado no diretório de cache.
    """
    logger.info(f"Iniciando o carregamento do tokenizer: {settings.tokenizer_name}")
    
    cache_dir = os.path.join(os.getcwd(), ".cache", "transformers")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Instancia e faz o download se não estiver em cache
        tokenizer = AutoTokenizer.from_pretrained(
            settings.tokenizer_name,
            revision=settings.tokenizer_revision or None,
            cache_dir=cache_dir
        )
        # Suprime os avisos de sequência longa definindo o tamanho máximo interno para um valor alto,
        # já que nós mesmos fazemos o chunking da janela de contexto no train.py
        tokenizer.model_max_length = int(1e9)
        logger.info("Tokenizer carregado com sucesso.")
        return tokenizer
    except Exception as e:
        logger.error(f"Erro ao carregar o tokenizer: {e}")
        raise

# Exporta a instância global do tokenizer
tokenizer = get_tokenizer()


@lru_cache(maxsize=4)
def _vocabulary_sha256(name_or_path: str, vocabulary_size: int) -> str:
    """Fingerprint the exact token-id mapping used by embeddings and logits."""
    digest = hashlib.sha256()
    digest.update(vocabulary_size.to_bytes(8, "little", signed=False))
    for token_id in range(vocabulary_size):
        token = tokenizer.convert_ids_to_tokens(token_id)
        digest.update(token_id.to_bytes(8, "little", signed=False))
        digest.update(str(token).encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


def tokenizer_metadata() -> dict[str, object]:
    """Return checkpoint metadata that uniquely identifies the token-id map."""
    vocabulary_size = int(tokenizer.vocab_size)
    name_or_path = str(tokenizer.name_or_path)
    return {
        "name_or_path": name_or_path,
        "configured_name": settings.tokenizer_name,
        "configured_revision": settings.tokenizer_revision or None,
        "class": tokenizer.__class__.__name__,
        "vocab_size": vocabulary_size,
        "vocab_sha256": _vocabulary_sha256(name_or_path, vocabulary_size),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }


def validate_tokenizer_metadata(
    checkpoint_config: dict,
    expected_vocab_size: int | None = None,
    *,
    allow_legacy: bool = True,
) -> None:
    """Reject a checkpoint whose token IDs do not mean the same thing locally."""
    current = tokenizer_metadata()
    if expected_vocab_size is not None and int(current["vocab_size"]) != int(expected_vocab_size):
        raise RuntimeError(
            "Tokenizer incompatível com os embeddings do checkpoint: "
            f"tokenizer={current['vocab_size']}, checkpoint={expected_vocab_size}."
        )

    saved = checkpoint_config.get("tokenizer")
    if not isinstance(saved, dict):
        if allow_legacy:
            logger.warning(
                "Checkpoint legado sem fingerprint do tokenizer; aceitando somente pela dimensão do vocabulário."
            )
            return
        raise RuntimeError("Checkpoint não registra a identidade do tokenizer.")

    mismatches = []
    for key in (
        "vocab_size",
        "vocab_sha256",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    ):
        if saved.get(key) != current.get(key):
            mismatches.append(f"{key}: checkpoint={saved.get(key)!r}, atual={current.get(key)!r}")
    if mismatches:
        raise RuntimeError("Tokenizer incompatível com o checkpoint: " + "; ".join(mismatches))


def decode_continuation(
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
) -> str:
    """Decode generated tokens in the context of their prompt.

    SentencePiece tokenizers encode a word boundary inside tokens such as
    ``▁white``.  Decoding that token by itself intentionally drops the leading
    boundary, while decoding it after the prompt produces the required space.
    """
    decode_options = {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    prompt_ids = [int(token_id) for token_id in prompt_token_ids]
    generated_ids = [int(token_id) for token_id in generated_token_ids]
    decoded_prompt = tokenizer.decode(prompt_ids, **decode_options)
    decoded_full = tokenizer.decode(prompt_ids + generated_ids, **decode_options)
    if decoded_full.startswith(decoded_prompt):
        return decoded_full[len(decoded_prompt) :]

    # Conservative fallback for tokenizers whose normalization can rewrite the
    # already-decoded prompt when more tokens are appended.
    return tokenizer.decode(generated_ids, **decode_options)
