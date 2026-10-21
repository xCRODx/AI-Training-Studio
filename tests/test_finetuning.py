import json
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from src.finetuning.backends import LocalPyTorchBackend, TorchTitanBackend
from src.finetuning.chat_format import ensure_system_message, format_generation_prompt
from src.finetuning.checkpoints import CSV_HEADERS, FineTuningCheckpointStore
from src.finetuning.config import FineTuningSettings
from src.finetuning.dataset import (
    _conversation_payload,
    _encode_conversation_samples,
    _encode_messages,
    _normalize_messages,
    _sample_stream,
    _select,
    ConversationTensorDataset,
    collate_conversation_batch,
)
from src.finetuning.manager import FineTuningManager
from src.finetuning.lora import adapter_state_dict, attach_lora, merged_state_dict
from src.finetuning.trainer import fine_tuning_recipe_id, resolve_fine_tuning_context
from src.model import CustomLLM
from src.tokenizer import tokenizer


class FineTuningCheckpointTests(unittest.TestCase):
    def test_bootstrap_copies_only_model_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "checkpoints" / "base"
            destination = root / "checkpoints_finetuning"
            source.mkdir(parents=True)
            save_file({"weight": torch.ones(2, 2)}, str(source / "model.safetensor"))
            (source / "config.json").write_text(json.dumps({"model_d_model": 2}), encoding="utf-8")
            torch.save({"pretrain": True}, source / "optimizer.pt")

            settings = FineTuningSettings(
                _env_file=None,
                source_dir=str(source),
                root_dir=str(destination),
            )
            store = FineTuningCheckpointStore(settings)
            self.assertTrue(store.bootstrap())
            self.assertTrue((settings.base_path / "model.safetensor").is_file())
            self.assertTrue((settings.base_path / "config.json").is_file())
            self.assertFalse((settings.base_path / "optimizer.pt").exists())
            header = (settings.base_path / "training_metrics.csv").read_text(encoding="utf-8").strip()
            self.assertEqual(header.split(","), CSV_HEADERS)
            self.assertFalse(store.bootstrap())

    def test_lora_profile_is_isolated_from_legacy_full_path(self):
        settings = FineTuningSettings(_env_file=None, root_dir="isolated-fine-tuning")
        lora = settings.model_copy(update={"method": "lora"})
        self.assertEqual(settings.base_path.name, "base_finetuning")
        self.assertEqual(lora.base_path.name, "base_finetuning_lora")
        self.assertNotEqual(settings.base_path, lora.base_path)


class NativeLoRATests(unittest.TestCase):
    def test_lora_freezes_base_and_preserves_standard_merged_keys(self):
        torch.manual_seed(3)
        model = CustomLLM(
            vocab_size=64, d_model=32, num_layers=1, num_heads=4, max_context=16
        )
        inputs = torch.randint(0, 64, (2, 6))
        baseline = model(inputs).detach()

        report = attach_lora(
            model,
            rank=4,
            alpha=8,
            dropout=0,
            target_modules=("q_proj", "v_proj"),
        )
        self.assertGreater(report.trainable_parameters, 0)
        self.assertLess(report.trainable_parameters, report.total_parameters)
        self.assertTrue(torch.allclose(baseline, model(inputs), atol=1e-6))
        self.assertTrue(adapter_state_dict(model))
        self.assertTrue(all(
            parameter.requires_grad == ("lora_adapter" in name)
            for name, parameter in model.named_parameters()
        ))

        merged = merged_state_dict(model)
        self.assertTrue(any(key.endswith("q_proj.weight") for key in merged))
        self.assertFalse(any("lora_adapter" in key for key in merged))

    def test_lora_checkpoint_publishes_adapter_and_merged_model(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = FineTuningSettings(
                _env_file=None,
                method="lora",
                root_dir=directory,
                lora_rank=2,
                lora_alpha=4,
                lora_dropout=0,
                lora_target_modules="q_proj,v_proj",
            )
            model = CustomLLM(
                vocab_size=64, d_model=32, num_layers=1, num_heads=4, max_context=16
            )
            settings.base_path.mkdir(parents=True)
            base_state = {
                key: value.detach().contiguous()
                for key, value in model.state_dict().items()
                if key != "lm_head.weight"
            }
            save_file(base_state, str(settings.base_model_path))
            attach_lora(
                model,
                rank=2,
                alpha=4,
                dropout=0,
                target_modules=("q_proj", "v_proj"),
            )
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

            FineTuningCheckpointStore(settings).save(
                model,
                optimizer,
                scheduler,
                {"fine_tuning_optimizer_step": 1},
            )

            self.assertTrue(settings.adapter_path.is_file())
            self.assertTrue((settings.base_path / "model.safetensor").is_file())
            adapter = load_file(str(settings.adapter_path))
            published = load_file(str(settings.base_path / "model.safetensor"))
            self.assertTrue(adapter)
            self.assertFalse(any("lora_adapter" in key for key in published))
            config = json.loads((settings.base_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["fine_tuning_method"], "lora")


class FineTuningDatasetTests(unittest.TestCase):
    def test_missing_system_prompt_is_injected_like_inference(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        normalized, injected = ensure_system_message(messages, "Be concise.")
        self.assertTrue(injected)
        self.assertEqual(normalized[0], {"role": "system", "content": "Be concise."})
        self.assertEqual(
            format_generation_prompt([messages[0]], "Be concise."),
            "System: Be concise.\nUser: Hello\nAssistant:",
        )

    def test_tucano_conversations_field_is_supported_explicitly(self):
        item = {
            "conversations": [
                {"role": "user", "content": "Olá"},
                {"role": "assistant", "content": "Olá!"},
            ]
        }
        self.assertEqual(_conversation_payload(item, "conversations"), item["conversations"])

    def test_assistant_only_mask(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        normalized = _normalize_messages(messages)
        self.assertIsNotNone(normalized)
        encoded = _encode_messages(normalized, tokenizer, max_length=32, assistant_only=True)
        self.assertIsNotNone(encoded)
        labels = encoded["labels"]
        self.assertTrue((labels == -100).any())
        self.assertGreater(int((labels != -100).sum()), 1)
        self.assertLessEqual(int(encoded["input_ids"].numel()), 32)

    def test_rejects_broken_role_order(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Anyone?"},
        ]
        self.assertIsNone(_normalize_messages(messages))

    def test_long_conversation_splits_only_between_complete_pairs(self):
        messages = [
            {"role": "user", "content": "Question number one please"},
            {"role": "assistant", "content": "This is a concise answer. " * 4},
            {"role": "user", "content": "Question number two please"},
            {"role": "assistant", "content": "This is another concise answer. " * 4},
        ]
        samples, stats = _encode_conversation_samples(
            messages, tokenizer, max_length=64, assistant_only=True
        )
        self.assertEqual(len(samples), 2)
        self.assertEqual(stats["pairs_preserved"], 2)
        self.assertEqual(stats["pairs_over_context"], 0)
        self.assertEqual(stats["truncated"], 0)
        self.assertEqual(stats["prompt_preserved_samples"], 2)
        self.assertTrue(all(int(sample["input_ids"].numel()) <= 64 for sample in samples))

    def test_ragged_cache_and_dynamic_batch_padding(self):
        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([-100, 2, 3]),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "supervised_tokens": torch.tensor(2),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([-100, 5]),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "supervised_tokens": torch.tensor(1),
            },
        ]
        dataset = ConversationTensorDataset.from_samples(samples)
        self.assertEqual(dataset.total_tokens, 5)
        batch = collate_conversation_batch([dataset[0], dataset[1]], pad_token_id=0)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["labels"][1].tolist(), [-100, 5, -100])

    def test_limited_selection_is_seeded_and_not_a_prefix(self):
        from datasets import Dataset

        dataset = Dataset.from_dict({"value": list(range(100))})
        first = _select(dataset, 10, seed=42)["value"]
        second = _select(dataset, 10, seed=42)["value"]
        self.assertEqual(first, second)
        self.assertNotEqual(first, list(range(10)))

    def test_stream_sample_materializes_only_the_requested_limit(self):
        from datasets import Dataset

        source = Dataset.from_dict({"value": list(range(100))}).to_iterable_dataset()
        sampled = _sample_stream(source, 12, seed=42)
        self.assertEqual(len(sampled), 12)
        self.assertNotEqual(sampled["value"], list(range(12)))


class PretrainingTemplateTests(unittest.TestCase):
    def test_template_preserves_every_legacy_javascript_id(self):
        root = Path(__file__).resolve().parents[1]
        javascript = (root / "static" / "script.js").read_text(encoding="utf-8")
        template = (root / "templates" / "index.html").read_text(encoding="utf-8")
        required = set(re.findall(r"getElementById\(['\"]([^'\"]+)", javascript))
        available = set(re.findall(r'id="([^"]+)"', template))
        self.assertEqual(required - available, set())

    def test_playground_template_preserves_javascript_ids(self):
        root = Path(__file__).resolve().parents[1]
        javascript = (root / "static" / "playground.js").read_text(encoding="utf-8")
        template = (root / "templates" / "playground.html").read_text(encoding="utf-8")
        required = set(re.findall(r"\$\(['\"]([^'\"]+)", javascript))
        available = set(re.findall(r'id="([^"]+)"', template))
        self.assertEqual(required - available, set())


class ModelFactoryTests(unittest.TestCase):
    def test_custom_architecture_values_do_not_change_legacy_constructor(self):
        model = CustomLLM(vocab_size=100, d_model=64, num_layers=1)
        output = model(torch.randint(0, 100, (2, 8)))
        self.assertEqual(tuple(output.shape), (2, 8, 100))
        self.assertEqual(model.d_model, 64)
        self.assertEqual(model.num_layers, 1)

    def test_forward_rejects_sequences_beyond_checkpoint_context_clearly(self):
        model = CustomLLM(vocab_size=100, d_model=64, num_layers=1, max_context=8)
        with self.assertRaisesRegex(ValueError, "limite=8"):
            model(torch.randint(0, 100, (1, 9)))

    def test_right_padding_cannot_change_real_token_logits(self):
        torch.manual_seed(9)
        model = CustomLLM(vocab_size=100, d_model=64, num_layers=1, max_context=8).eval()
        short = torch.tensor([[7, 8, 9]])
        padded_batch = torch.tensor([[7, 8, 9, 0, 0], [3, 4, 5, 6, 7]])
        with torch.inference_mode():
            short_logits = model(short)
            batch_logits = model(padded_batch)
        self.assertTrue(torch.allclose(short_logits, batch_logits[:1, :3], atol=1e-6))


class FineTuningContextTests(unittest.TestCase):
    def test_zero_context_uses_checkpoint_limit_automatically(self):
        self.assertEqual(resolve_fine_tuning_context(0, {"context_window": 8192}), 8192)

    def test_requested_context_is_clamped_to_checkpoint_rope_limit(self):
        self.assertEqual(resolve_fine_tuning_context(512, {"context_window": 256}), 256)

    def test_shorter_requested_context_is_preserved(self):
        self.assertEqual(resolve_fine_tuning_context(128, {"context_window": 256}), 128)

    def test_missing_checkpoint_context_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "nao registra"):
            resolve_fine_tuning_context(512, {})

    def test_recipe_fingerprint_changes_with_dataset(self):
        first = FineTuningSettings(_env_file=None, dataset_name="dataset-a")
        second = first.model_copy(update={"dataset_name": "dataset-b"})
        self.assertNotEqual(
            fine_tuning_recipe_id(first, 256, 1000),
            fine_tuning_recipe_id(second, 256, 1000),
        )


class BackendContractTests(unittest.TestCase):
    def test_backend_capabilities_are_stable(self):
        self.assertFalse(LocalPyTorchBackend().capabilities.distributed)
        self.assertTrue(TorchTitanBackend().capabilities.fsdp2)
        self.assertTrue(TorchTitanBackend().capabilities.distributed_checkpoint)

    def test_manager_launches_backend_without_touching_pretrain_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "base"
            source.mkdir()
            save_file({"weight": torch.ones(2, 2)}, str(source / "model.safetensor"))
            (source / "config.json").write_text(
                json.dumps({"model_d_model": 2, "model_layers": 1, "context_window": 8}),
                encoding="utf-8",
            )
            settings = FineTuningSettings(
                _env_file=None,
                source_dir=str(source),
                root_dir=str(root / "fine"),
                context_window=8,
            )
            manager = FineTuningManager(settings)
            with patch("src.finetuning.trainer.LocalFineTuningTrainer.run", return_value="completed"):
                manager.start()
                for _ in range(100):
                    if manager.status()["state"] == "completed":
                        break
                    time.sleep(0.01)
            self.assertEqual(manager.status()["state"], "completed")
            self.assertFalse((settings.base_path / "optimizer.pt").exists())


if __name__ == "__main__":
    unittest.main()
