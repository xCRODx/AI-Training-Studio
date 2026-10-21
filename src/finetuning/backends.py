import threading
from dataclasses import asdict, dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    distributed: bool
    fsdp2: bool
    tensor_parallel: bool
    pipeline_parallel: bool
    distributed_checkpoint: bool
    torch_compile: bool


class TrainingBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def launch(self, target: Callable[[], None]) -> threading.Thread:
        ...

    def describe(self) -> dict[str, object]:
        ...


class LocalPyTorchBackend:
    name = "local"
    capabilities = BackendCapabilities(
        distributed=False,
        fsdp2=False,
        tensor_parallel=False,
        pipeline_parallel=False,
        distributed_checkpoint=False,
        torch_compile=True,
    )

    def launch(self, target: Callable[[], None]) -> threading.Thread:
        worker = threading.Thread(target=target, name="fine-tuning-local-worker", daemon=True)
        worker.start()
        return worker

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "available": True, "capabilities": asdict(self.capabilities)}


class TorchTitanBackend:
    """Stable control-plane contract for the future TorchTitan worker."""

    name = "torchtitan"
    capabilities = BackendCapabilities(
        distributed=True,
        fsdp2=True,
        tensor_parallel=True,
        pipeline_parallel=True,
        distributed_checkpoint=True,
        torch_compile=True,
    )

    def launch(self, target: Callable[[], None]) -> threading.Thread:
        raise RuntimeError(
            "O backend TorchTitan ainda não foi instalado. Use FINE_TUNING_ENGINE=local nesta etapa."
        )

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "available": False, "capabilities": asdict(self.capabilities)}


def get_training_backend(name: str) -> TrainingBackend:
    if name == "local":
        return LocalPyTorchBackend()
    if name == "torchtitan":
        return TorchTitanBackend()
    raise ValueError(f"Backend de fine-tuning desconhecido: {name}")
