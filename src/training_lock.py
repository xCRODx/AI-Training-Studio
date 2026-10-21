from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path

from src.paths import PROJECT_ROOT


_PROCESS_TOKEN = uuid.uuid4().hex


class TrainingLease:
    """Cross-process lease shared by Windows and the Docker bind mount."""

    def __init__(
        self,
        path: Path | None = None,
        heartbeat_seconds: float = 10.0,
        stale_seconds: float = 90.0,
    ):
        self.path = (path or PROJECT_ROOT / ".retbit-training.lock").resolve()
        self.heartbeat_seconds = heartbeat_seconds
        self.stale_seconds = stale_seconds
        self._owned = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._purpose = ""

    @property
    def owned(self) -> bool:
        return self._owned

    def _owner_payload(self, purpose: str) -> str:
        return json.dumps(
            {
                "purpose": purpose,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "process_token": _PROCESS_TOKEN,
                "started_at": time.time(),
            },
            ensure_ascii=False,
        )

    def describe_owner(self) -> str:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return f"{payload.get('purpose', 'treino')} em {payload.get('host', '?')} (PID {payload.get('pid', '?')})"
        except (OSError, ValueError, json.JSONDecodeError):
            return str(self.path)

    def is_available(self) -> bool:
        if self._owned or not self.path.exists():
            return True
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            same_process_slot = (
                payload.get("host") == socket.gethostname()
                and int(payload.get("pid", -1)) == os.getpid()
            )
            if same_process_slot and payload.get("process_token") != _PROCESS_TOKEN:
                # Containers podem reciclar hostname e PID após um restart. O token
                # distingue a nova instância de duas leases vivas no mesmo processo.
                return True
            return time.time() - self.path.stat().st_mtime > self.stale_seconds
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def acquire(self, purpose: str) -> None:
        if self._owned:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(self._owner_payload(purpose))
                self._owned = True
                self._purpose = purpose
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._heartbeat,
                    name="retbit-training-lease",
                    daemon=True,
                )
                self._thread.start()
                return
            except FileExistsError:
                if not self.is_available():
                    raise RuntimeError(
                        "Outro processo já possui o checkpoint: " + self.describe_owner()
                    )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        raise RuntimeError("Não foi possível adquirir a trava exclusiva de treinamento.")

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            if not self._owned:
                return
            try:
                os.utime(self.path, None)
            except OSError:
                return

    def release(self) -> None:
        if not self._owned:
            return
        self._owned = False
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


training_lease = TrainingLease()
