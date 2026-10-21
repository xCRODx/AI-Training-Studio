import sys
import os
from dotenv import load_dotenv

# Força a pasta raiz do projeto no topo da memória do Python
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Carrega as variáveis do .env.local na memória global ANTES de qualquer importação
load_dotenv(".env.local")

import uvicorn
# Importa o app diretamente como um objeto nativo, ignorando o parser de texto falho do Uvicorn
from src.main import app, state
from src.lifecycle import shutdown_requested


class RetBitServer(uvicorn.Server):
    """Solicita checkpoint antes que o Uvicorn espere as tasks de background."""

    def handle_exit(self, sig, frame):
        shutdown_requested.set()
        if state.training_task is not None and not state.training_task.done():
            if state.is_training.get("active", False):
                state.is_training["checkpoint_saved"] = False
                state.is_training["pause_requested"] = True
            state.is_training["shutdown_requested"] = True
        super().handle_exit(sig, frame)

if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_graceful_shutdown=150,
    )
    RetBitServer(config).run()
