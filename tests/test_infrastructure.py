import tempfile
import unittest
import json
import os
import socket
from pathlib import Path
from unittest.mock import patch

from src.paths import (
    resolve_finetuning_root_dir,
    resolve_finetuning_source_dir,
    resolve_pretraining_checkpoint_dir,
)
from src.training_lock import TrainingLease
from src import train as train_module


class SharedPathResolutionTests(unittest.TestCase):
    def test_empty_checkpoint_always_uses_organized_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.safetensor", "optimizer.pt", "config.json"):
                (root / name).write_bytes(b"checkpoint")
            default = root / "checkpoints" / "base"
            with (
                patch("src.paths.PROJECT_ROOT", root),
                patch("src.paths.DEFAULT_PRETRAINING_CHECKPOINT_DIR", default),
            ):
                self.assertEqual(resolve_pretraining_checkpoint_dir(""), default.resolve())
                self.assertEqual(resolve_finetuning_source_dir("", ""), default.resolve())

    def test_empty_paths_use_organized_defaults_for_new_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = root / "checkpoints" / "base"
            with (
                patch("src.paths.PROJECT_ROOT", root),
                patch("src.paths.DEFAULT_PRETRAINING_CHECKPOINT_DIR", default),
                patch("src.paths.DEFAULT_FINETUNING_ROOT_DIR", root / "checkpoints_finetuning"),
            ):
                self.assertEqual(resolve_pretraining_checkpoint_dir(""), default.resolve())
                self.assertEqual(
                    resolve_finetuning_root_dir(""),
                    (root / "checkpoints_finetuning").resolve(),
                )


class TrainingLeaseTests(unittest.TestCase):
    def test_second_process_cannot_claim_live_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".training.lock"
            first = TrainingLease(path, heartbeat_seconds=0.01, stale_seconds=1)
            second = TrainingLease(path, heartbeat_seconds=0.01, stale_seconds=1)
            try:
                first.acquire("pré-treino")
                with self.assertRaisesRegex(RuntimeError, "Outro processo"):
                    second.acquire("fine-tuning")
            finally:
                first.release()
                second.release()
            self.assertFalse(path.exists())

    def test_recycled_container_pid_can_replace_an_old_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".training.lock"
            path.write_text(
                json.dumps({
                    "purpose": "pré-treino",
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "process_token": "old-container-instance",
                }),
                encoding="utf-8",
            )
            lease = TrainingLease(path, stale_seconds=3600)
            try:
                lease.acquire("pré-treino")
                self.assertTrue(lease.owned)
            finally:
                lease.release()


class PersistentSnapshotTests(unittest.TestCase):
    def test_legacy_root_snapshot_never_copies_the_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.safetensor", "optimizer.pt", "config.json", "training_metrics.csv"):
                (root / name).write_bytes(b"checkpoint")
            (root / "source-code-must-not-be-copied.py").write_text("secret", encoding="utf-8")
            with (
                patch.object(train_module, "CHECKPOINT_DIR", str(root)),
                patch.object(train_module, "PROJECT_ROOT", root),
            ):
                snapshot = train_module.create_persistent_snapshot(123)

            self.assertEqual(snapshot, root / "checkpoints" / "checkpoint_123")
            self.assertTrue((snapshot / "model.safetensor").is_file())
            self.assertTrue((snapshot / "optimizer.pt").is_file())
            self.assertTrue((snapshot / "config.json").is_file())
            self.assertFalse((snapshot / "source-code-must-not-be-copied.py").exists())


if __name__ == "__main__":
    unittest.main()
