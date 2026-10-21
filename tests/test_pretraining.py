import asyncio
import copy
import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import gguf
from safetensors.torch import save_file

import src.main as main_module
from src.architecture import Architecture, resolve_architecture
from src.base_inference import BaseCheckpointInference
from src.compile_runtime.converter import _ternary, convert_checkpoint_to_gguf
from src.generation import advance_generation, prefill_generation
from src.model import CustomLLM
from src.tokenizer import (
    decode_continuation,
    tokenizer,
    tokenizer_metadata,
    validate_tokenizer_metadata,
)
import src.train as train_module
from src.sampling import resolve_sampling


class ArchitectureResolutionTests(unittest.TestCase):
    def test_flag_matrix_resolves_correct_architectures(self):
        self.assertEqual(resolve_architecture(False, False), Architecture.TRANSFORMER)
        self.assertEqual(resolve_architecture(True, False), Architecture.BITNET)
        self.assertEqual(resolve_architecture(False, True), Architecture.RETNET)
        self.assertEqual(resolve_architecture(True, True), Architecture.RETBIT)

    def test_flags_default_to_false_in_settings(self):
        from src.config import Settings

        defaults = Settings(_env_file=None)
        self.assertFalse(defaults.use_bitnet)
        self.assertFalse(defaults.use_retnet)
        self.assertEqual(defaults.architecture, "transformer")


class ModelArchitectureTests(unittest.TestCase):
    def _assert_forward_and_recurrent_agree(self, architecture):
        torch.manual_seed(0)
        model = CustomLLM(
            vocab_size=200,
            d_model=64,
            num_layers=1,
            architecture=architecture,
            max_context=32,
        )
        model.eval()
        ids = torch.randint(0, 200, (1, 7))
        with torch.no_grad():
            logits = model(ids)
            self.assertEqual(tuple(logits.shape), (1, 7, 200))
            logits_pre, states = model.prefill(ids)
            self.assertEqual(tuple(logits_pre.shape), (1, 200))
            next_token = torch.tensor([[42]])
            logits_rec, states = model.recurrent(next_token, states)
            # compara com um prefill completo de 8 tokens
            full = torch.cat([ids, next_token], dim=-1)
            logits_full, _ = model.prefill(full)
            delta = (logits_rec - logits_full).abs().max().item()
            self.assertLess(delta, 1e-3)

    def test_transformer_kv_cache_prefill_recurrent_agree(self):
        self._assert_forward_and_recurrent_agree(Architecture.TRANSFORMER)

    def test_bitnet_kv_cache_prefill_recurrent_agree(self):
        self._assert_forward_and_recurrent_agree(Architecture.BITNET)

    def test_retnet_prefill_recurrent_agree(self):
        self._assert_forward_and_recurrent_agree(Architecture.RETNET)

    def test_retbit_keeps_legacy_state_dict_keys(self):
        torch.manual_seed(0)
        model = CustomLLM(vocab_size=200, d_model=64, num_layers=1, architecture=Architecture.RETBIT)
        keys = set(model.state_dict().keys())
        self.assertIn("layers.0.retention.q_proj.weight", keys)
        self.assertIn("layers.0.retention.gammas", keys)
        self.assertIn("layers.0.ffn.fc1.weight", keys)
        self.assertIn("layers.0.norm1.weight", keys)
        self.assertNotIn("layers.0.attention.q_proj.weight", keys)

    def test_transformer_uses_attention_keys(self):
        torch.manual_seed(0)
        model = CustomLLM(vocab_size=200, d_model=64, num_layers=1, architecture=Architecture.TRANSFORMER)
        keys = set(model.state_dict().keys())
        self.assertIn("layers.0.attention.q_proj.weight", keys)
        self.assertIn("layers.0.ffn.gate_proj.weight", keys)
        self.assertIn("layers.0.ffn.up_proj.weight", keys)
        self.assertIn("layers.0.ffn.down_proj.weight", keys)

    def test_bitnet_quantizes_but_transformer_does_not(self):
        from src.model import BitLinear

        bitnet = CustomLLM(vocab_size=200, d_model=64, num_layers=1, architecture=Architecture.BITNET)
        transformer = CustomLLM(vocab_size=200, d_model=64, num_layers=1, architecture=Architecture.TRANSFORMER)
        bitnet_quantized = sum(m.quantized for m in bitnet.modules() if isinstance(m, BitLinear))
        transformer_quantized = sum(m.quantized for m in transformer.modules() if isinstance(m, BitLinear))
        self.assertGreater(bitnet_quantized, 0)
        self.assertEqual(transformer_quantized, 0)


class LanguageModelInvariantTests(unittest.TestCase):
    def test_next_token_shift_maps_each_logit_to_the_following_token(self):
        batch = torch.tensor([[10, 20, 30, 40]])
        inputs = batch[:, :-1]
        targets = batch[:, 1:].reshape(-1)
        logits = torch.full((1, 3, 50), -10.0)
        for position, target in enumerate(targets):
            logits[0, position, target] = 10.0

        shifted_loss = F.cross_entropy(logits.reshape(-1, 50), targets)
        unshifted_loss = F.cross_entropy(logits.reshape(-1, 50), inputs.reshape(-1))

        self.assertEqual(inputs.tolist(), [[10, 20, 30]])
        self.assertEqual(targets.tolist(), [20, 30, 40])
        self.assertLess(shifted_loss.item(), 1e-6)
        self.assertGreater(unshifted_loss.item(), 10.0)

    def test_future_tokens_cannot_change_earlier_logits(self):
        for architecture in Architecture:
            with self.subTest(architecture=architecture.value):
                torch.manual_seed(7)
                model = CustomLLM(
                    vocab_size=97,
                    d_model=64,
                    num_layers=2,
                    architecture=architecture,
                    max_context=32,
                ).eval()
                original = torch.tensor([[3, 11, 25, 40, 7, 9]])
                changed = original.clone()
                changed[0, 4:] = torch.tensor([88, 89])

                with torch.no_grad():
                    original_logits = model(original)
                    changed_logits = model(changed)

                self.assertEqual(
                    (original_logits[:, :4] - changed_logits[:, :4]).abs().max().item(),
                    0.0,
                )

    def test_incremental_logits_match_every_full_prefix(self):
        for architecture in Architecture:
            with self.subTest(architecture=architecture.value):
                torch.manual_seed(9)
                model = CustomLLM(
                    vocab_size=97,
                    d_model=64,
                    num_layers=1,
                    architecture=architecture,
                    max_context=32,
                ).eval()
                token_ids = torch.tensor([[3, 11, 25, 40, 7, 9]])

                with torch.no_grad():
                    incremental, states = model.prefill(token_ids[:, :1])
                    full = model(token_ids[:, :1])[:, -1]
                    self.assertTrue(torch.allclose(incremental, full, atol=1e-5, rtol=1e-5))
                    for length in range(2, token_ids.size(1) + 1):
                        incremental, states = model.recurrent(token_ids[:, length - 1 : length], states)
                        full = model(token_ids[:, :length])[:, -1]
                        self.assertTrue(torch.allclose(incremental, full, atol=1e-5, rtol=1e-5))
                        self.assertEqual(incremental.argmax(-1).item(), full.argmax(-1).item())

    def test_output_index_uses_the_matching_embedding_row_and_greedy_uses_argmax(self):
        torch.manual_seed(11)
        model = CustomLLM(vocab_size=97, d_model=64, num_layers=1).eval()
        token_ids = torch.tensor([[3, 11, 25]])
        with torch.no_grad():
            hidden = model.embeddings(token_ids)
            for layer in model.layers:
                hidden = layer(hidden)
            hidden = model.norm_f(hidden)
            logits = model(token_ids)
            manual_logits = F.linear(hidden, model.embeddings.weight)

        self.assertIs(model.lm_head.weight, model.embeddings.weight)
        self.assertTrue(torch.equal(logits, manual_logits))
        self.assertEqual(logits[:, -1].argmax(-1).item(), manual_logits[:, -1].argmax(-1).item())

    def test_context_rollover_is_safe_for_every_architecture(self):
        for architecture in Architecture:
            with self.subTest(architecture=architecture.value):
                torch.manual_seed(13)
                model = CustomLLM(
                    vocab_size=97,
                    d_model=64,
                    num_layers=1,
                    architecture=architecture,
                    max_context=8,
                ).eval()
                prompt = torch.tensor([[3, 11, 25, 40, 7, 9, 12]])
                visible, _, states = prefill_generation(model, prompt, 8)
                visible, _, states, rolled = advance_generation(
                    model, visible, torch.tensor([[15]]), states, 8
                )
                self.assertFalse(rolled)
                visible, cached, states, rolled = advance_generation(
                    model, visible, torch.tensor([[16]]), states, 8
                )
                self.assertTrue(rolled)
                with torch.no_grad():
                    full = model(visible)[:, -1]
                self.assertTrue(torch.allclose(cached, full, atol=1e-5, rtol=1e-5))


class BaseCheckpointInferenceTests(unittest.TestCase):
    def test_loads_checkpoint_without_starting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            model = CustomLLM(vocab_size=tokenizer.vocab_size, d_model=64, num_layers=1)
            state = model.state_dict()
            state.pop("lm_head.weight", None)
            save_file(state, str(checkpoint / "model.safetensor"))
            (checkpoint / "config.json").write_text(
                json.dumps({"model_d_model": 64, "model_layers": 1, "context_window": 32}),
                encoding="utf-8",
            )

            service = BaseCheckpointInference(checkpoint, context_window=32)
            loaded = service.load("cpu")

            self.assertFalse(loaded.training)
            self.assertEqual(loaded.d_model, 64)
            self.assertEqual(loaded.num_layers, 1)
            self.assertIs(service.load("cpu"), loaded)


class TokenizerContinuationTests(unittest.TestCase):
    def test_sentencepiece_word_boundary_is_preserved_after_prompt(self):
        prompt = "the ball is"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = tokenizer.encode(" white", add_special_tokens=False)

        self.assertEqual(decode_continuation(prompt_ids, generated_ids), " white")
        self.assertEqual(prompt + decode_continuation(prompt_ids, generated_ids), "the ball is white")

    def test_punctuation_is_not_prefixed_with_a_space(self):
        prompt_ids = tokenizer.encode("the ball is", add_special_tokens=False)
        comma_id = tokenizer.convert_tokens_to_ids(",")

        self.assertEqual(decode_continuation(prompt_ids, [comma_id]), ",")

    def test_checkpoint_fingerprint_rejects_a_different_vocabulary(self):
        metadata = tokenizer_metadata()
        validate_tokenizer_metadata({"tokenizer": metadata}, tokenizer.vocab_size)
        altered = dict(metadata)
        altered["vocab_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "Tokenizer incompatível"):
            validate_tokenizer_metadata({"tokenizer": altered}, tokenizer.vocab_size)

    def test_portuguese_unicode_roundtrip_has_no_unknown_tokens(self):
        text = "Ação, coração, informação, português, açúcar — € 😀"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertNotIn(tokenizer.unk_token_id, token_ids)
        self.assertEqual(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            text,
        )


class SamplingPolicyTests(unittest.TestCase):
    def test_temperature_zero_in_auto_mode_uses_configured_policy(self):
        policy = resolve_sampling(
            "auto", temperature=0, top_p=1, top_k=1, repetition_penalty=1,
            auto_temperature=0.7, auto_top_p=0.9, auto_top_k=50,
            auto_repetition_penalty=1.05,
        )
        self.assertEqual(policy.mode, "sample")
        self.assertEqual(policy.temperature, 0.7)
        self.assertEqual(policy.top_k, 50)

    def test_greedy_is_an_explicit_mode(self):
        policy = resolve_sampling(
            "greedy", temperature=1, top_p=0.5, top_k=50, repetition_penalty=1.2,
            auto_temperature=0.7, auto_top_p=0.9, auto_top_k=50,
            auto_repetition_penalty=1.05,
        )
        self.assertEqual(policy.mode, "greedy")
        self.assertEqual(policy.temperature, 0)


class PretrainingDatasetInvariantTests(unittest.TestCase):
    class _Dataset:
        def __init__(self, texts):
            self.items = [{"text": text} for text in texts]

        def __iter__(self):
            return iter(self.items)

        def state_dict(self):
            return {"position": 0}

        def load_state_dict(self, state):
            return None

    class _StatefulDataset:
        def __init__(self, texts, fail_restore=False):
            self.items = [{"text": text} for text in texts]
            self.position = 0
            self.fail_restore = fail_restore

        def __iter__(self):
            while self.position < len(self.items):
                item = self.items[self.position]
                self.position += 1
                yield item

        def state_dict(self):
            return {"position": self.position}

        def load_state_dict(self, state):
            if self.fail_restore:
                raise RuntimeError("restore failed")
            self.position = int(state["position"])

        def skip(self, count):
            self.position += int(count)
            return self

    def _text_for_partition(self, validation: bool) -> str:
        for index in range(10000):
            text = f"partition candidate {index} with enough words for a causal language model"
            if train_module._is_validation_document(text) is validation:
                return text
        self.fail("Não foi possível encontrar texto para a partição determinística")

    def test_documents_receive_eos_and_validation_never_becomes_a_train_batch(self):
        validation_text = self._text_for_partition(True)
        training_text = self._text_for_partition(False)
        state = {}
        dataset = self._Dataset([validation_text] * 3 + [training_text] * 3)
        with (
            patch.object(train_module, "load_dataset", return_value=dataset),
            patch.object(train_module.settings, "train_batch_size", 1),
            patch.object(train_module.settings, "overfit_mode", False),
            patch.object(train_module.settings, "dataset_quality_field", ""),
        ):
            generator = train_module.data_generator(tokenizer, state=state, context_window=16)
            train_batch = next(generator)

        self.assertTrue(state["validation_batches"])
        validation_ids = {
            token_id
            for batch in state["validation_batches"]
            for chunk in batch
            for token_id in chunk
        }
        self.assertNotEqual(train_batch[0].tolist(), state["validation_batches"][0][0])
        self.assertIn(tokenizer.eos_token_id, validation_ids | set(train_batch[0].tolist()))

    def test_resume_continues_with_the_exact_next_batch(self):
        texts = [f"document {index} with unique content for exact resume testing" for index in range(30)]
        datasets = []

        def factory(**_):
            dataset = self._StatefulDataset(texts)
            datasets.append(dataset)
            return dataset

        state = {"dataset_offset_authoritative": True}
        with (
            patch.object(train_module, "load_dataset", side_effect=factory),
            patch.object(train_module.settings, "train_batch_size", 1),
            patch.object(train_module.settings, "overfit_mode", False),
            patch.object(train_module.settings, "dataset_quality_field", ""),
            patch.object(train_module.settings, "validation_holdout_fraction", 0.0),
        ):
            generator = train_module.data_generator(tokenizer, state=state, context_window=16)
            first = next(generator)
            resume_state = {
                "state_dict": copy.deepcopy(state["dataset"].state_dict()),
                "dataset_offset": state["dataset_offset"],
                "dataset_offset_authoritative": True,
                "token_buffer": list(state.get("token_buffer", [])),
                "batch_buffer": copy.deepcopy(state.get("batch_buffer", [])),
            }
            expected_next = next(generator)
            resumed = train_module.data_generator(tokenizer, state=resume_state, context_window=16)
            actual_next = next(resumed)

        self.assertFalse(torch.equal(first, expected_next))
        self.assertTrue(torch.equal(expected_next, actual_next))
        self.assertGreater(resume_state["dataset_offset"], 0)

    def test_failed_state_restore_never_restarts_from_the_beginning(self):
        dataset = self._StatefulDataset(["enough unique words for one training batch"] * 10, fail_restore=True)
        state = {
            "state_dict": {"position": 5},
            "dataset_offset": 0,
            "dataset_offset_authoritative": False,
        }
        with (
            patch.object(train_module, "load_dataset", return_value=dataset),
            patch.object(train_module.settings, "overfit_mode", False),
        ):
            generator = train_module.data_generator(tokenizer, state=state, context_window=8)
            with self.assertRaises(train_module.DatasetPositionError):
                next(generator)


class StrictCheckpointTests(unittest.TestCase):
    def test_accumulation_restarts_with_a_full_window_after_process_restart(self):
        micro_step = 0
        boundaries = []
        for resumed_batch in range(1, 11):
            micro_step, boundary = train_module._advance_accumulation(micro_step, 10)
            if boundary:
                boundaries.append(resumed_batch)
        self.assertEqual(boundaries, [10])
        self.assertEqual(micro_step, 0)

    def test_checkpoint_refuses_partial_gradient_accumulation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = CustomLLM(
                vocab_size=tokenizer.vocab_size,
                d_model=64,
                num_layers=1,
                max_context=16,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            with (
                patch.object(train_module, "CHECKPOINT_DIR", str(root)),
                patch.object(train_module, "CHECKPOINT_MODEL", str(root / "model.safetensor")),
                patch.object(train_module, "CHECKPOINT_OPT", str(root / "optimizer.pt")),
                patch.object(train_module, "CHECKPOINT_CONFIG", str(root / "config.json")),
            ):
                with self.assertRaisesRegex(RuntimeError, "fronteira segura"):
                    train_module.save_checkpoint(
                        model,
                        optimizer,
                        5,
                        80,
                        {"accumulation_micro_step": 5},
                    )
            self.assertFalse((root / "model.safetensor").exists())

    def test_save_records_the_effective_model_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = CustomLLM(
                vocab_size=tokenizer.vocab_size,
                d_model=64,
                num_layers=1,
                architecture=Architecture.RETNET,
                max_context=16,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            with (
                patch.object(train_module, "CHECKPOINT_DIR", str(root)),
                patch.object(train_module, "CHECKPOINT_MODEL", str(root / "model.safetensor")),
                patch.object(train_module, "CHECKPOINT_OPT", str(root / "optimizer.pt")),
                patch.object(train_module, "CHECKPOINT_CONFIG", str(root / "config.json")),
            ):
                train_module.save_checkpoint(model, optimizer, 3, 48, {})

            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["architecture"], "retnet")
            self.assertEqual(config["model_d_model"], 64)
            self.assertEqual(config["context_window"], 16)
            self.assertEqual(config["tokenizer"]["vocab_sha256"], tokenizer_metadata()["vocab_sha256"])

    def test_load_rejects_missing_model_weights_instead_of_restarting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = CustomLLM(tokenizer.vocab_size, d_model=64, num_layers=1, max_context=16)
            state = source.state_dict()
            state.pop("lm_head.weight", None)
            state.pop("layers.0.norm1.weight")
            save_file(state, str(root / "model.safetensor"))
            optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
            torch.save(optimizer.state_dict(), root / "optimizer.pt")
            (root / "config.json").write_text(
                json.dumps({"tokenizer": tokenizer_metadata()}), encoding="utf-8"
            )
            target = CustomLLM(tokenizer.vocab_size, d_model=64, num_layers=1, max_context=16)
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
            with (
                patch.object(train_module, "CHECKPOINT_MODEL", str(root / "model.safetensor")),
                patch.object(train_module, "CHECKPOINT_OPT", str(root / "optimizer.pt")),
                patch.object(train_module, "CHECKPOINT_CONFIG", str(root / "config.json")),
            ):
                with self.assertRaisesRegex(RuntimeError, "Pesos incompatíveis"):
                    train_module.load_checkpoint(target, target_optimizer)

    def test_compilation_cannot_override_checkpoint_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = CustomLLM(
                tokenizer.vocab_size,
                d_model=64,
                num_layers=1,
                architecture=Architecture.TRANSFORMER,
                max_context=16,
            )
            state = model.state_dict()
            state.pop("lm_head.weight", None)
            save_file(state, str(root / "model.safetensor"))
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_d_model": 64,
                        "model_layers": 1,
                        "context_window": 16,
                        "architecture": "transformer",
                        "tokenizer": tokenizer_metadata(),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "não corresponde ao checkpoint"):
                convert_checkpoint_to_gguf(
                    root,
                    root / "wrong.gguf",
                    architecture=Architecture.BITNET,
                )

    def test_transformer_export_uses_llama_gguf_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = CustomLLM(
                tokenizer.vocab_size,
                d_model=64,
                num_layers=1,
                architecture=Architecture.TRANSFORMER,
                num_heads=1,
                head_dim=64,
                num_kv_heads=1,
                ffn_hidden_size=128,
                max_context=32,
            )
            state = model.state_dict()
            state.pop("lm_head.weight", None)
            save_file(state, str(root / "model.safetensor"))
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_d_model": 64,
                        "model_layers": 1,
                        "context_window": 32,
                        "architecture": "transformer",
                        "num_heads": 1,
                        "head_dim": 64,
                        "num_kv_heads": 1,
                        "ffn_hidden_size": 128,
                        "norm_eps": 1e-6,
                        "tokenizer": tokenizer_metadata(),
                    }
                ),
                encoding="utf-8",
            )
            output = root / "model.gguf"
            report = convert_checkpoint_to_gguf(root, output)
            reader = gguf.GGUFReader(output)

            architecture = reader.fields["general.architecture"]
            architecture_bytes = architecture.parts[architecture.data[0]].tolist()
            self.assertEqual(bytes(architecture_bytes).decode("utf-8"), "llama")
            self.assertIn("llama.attention.layer_norm_rms_epsilon", reader.fields)
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(
                tensors["output_norm.weight"].tensor_type,
                gguf.GGMLQuantizationType.F32,
            )
            self.assertEqual(report["target"], "llama.cpp")
            del tensors, architecture, reader
            gc.collect()

    def test_bitnet_export_preserves_local_graph_for_official_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = CustomLLM(
                tokenizer.vocab_size,
                d_model=64,
                num_layers=1,
                architecture=Architecture.BITNET,
                num_heads=1,
                head_dim=64,
                num_kv_heads=1,
                ffn_hidden_size=128,
                max_context=32,
            )
            state = model.state_dict()
            state.pop("lm_head.weight", None)
            save_file(state, str(root / "model.safetensor"))
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_d_model": 64,
                        "model_layers": 1,
                        "context_window": 32,
                        "architecture": "bitnet",
                        "num_heads": 1,
                        "head_dim": 64,
                        "num_kv_heads": 1,
                        "ffn_hidden_size": 128,
                        "norm_eps": 1e-6,
                        "tokenizer": tokenizer_metadata(),
                    }
                ),
                encoding="utf-8",
            )
            output = root / "model.gguf"
            report = convert_checkpoint_to_gguf(root, output)
            reader = gguf.GGUFReader(output)
            architecture = reader.fields["general.architecture"]
            architecture_bytes = architecture.parts[architecture.data[0]].tolist()
            tensor_names = {tensor.name for tensor in reader.tensors}

            self.assertEqual(bytes(architecture_bytes).decode("utf-8"), "llama")
            self.assertEqual(report["architecture"], "bitnet")
            self.assertEqual(report["target"], "bitnet.cpp")
            self.assertEqual(report["encoding"], "materialized-ternary-bf16")
            self.assertFalse(any(name.endswith(".scale") for name in tensor_names))
            del tensor_names, architecture, reader
            gc.collect()


class RetBitPackingTests(unittest.TestCase):
    def test_four_ternary_weights_are_packed_in_one_byte(self):
        packed, scale = _ternary(torch.tensor([-1.0, 0.0, 1.0, 0.0]))
        self.assertEqual(packed.nbytes, 1)
        self.assertEqual(int(packed.view("uint8")[0]), 100)
        self.assertAlmostEqual(float(scale[0]), 0.5)


class PretrainingControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_task = main_module.state.training_task
        self.previous_state = dict(main_module.state.is_training)
        self.paused_task = asyncio.create_task(asyncio.Event().wait())
        main_module.state.training_task = self.paused_task
        main_module.state.is_training["active"] = False
        main_module.state.is_training["pause_requested"] = False
        main_module.state.is_training["checkpoint_saved"] = True

    async def asyncTearDown(self):
        self.paused_task.cancel()
        try:
            await self.paused_task
        except asyncio.CancelledError:
            pass
        main_module.state.training_task = self.previous_task
        main_module.state.is_training.clear()
        main_module.state.is_training.update(self.previous_state)

    async def test_resume_reuses_existing_training_task(self):
        with patch.object(main_module.finetuning_manager, "is_active", return_value=False):
            response = await main_module.resume_training()

        self.assertEqual(response["state"], "running")
        self.assertIs(main_module.state.training_task, self.paused_task)

    async def test_status_exposes_paused_state(self):
        response = await main_module.training_status()
        self.assertEqual(response["state"], "paused")
        self.assertFalse(response["active"])

    async def test_pause_returns_only_after_checkpoint_confirmation(self):
        main_module.state.is_training["active"] = True

        async def finish_safe_pause():
            await asyncio.sleep(0.02)
            main_module.state.is_training["active"] = False
            main_module.state.is_training["pause_requested"] = False
            main_module.state.is_training["checkpoint_saved"] = True

        completion = asyncio.create_task(finish_safe_pause())
        response = await main_module.stop_training()
        await completion

        self.assertEqual(response["state"], "paused")
        self.assertTrue(response["checkpoint_saved"])


if __name__ == "__main__":
    unittest.main()
