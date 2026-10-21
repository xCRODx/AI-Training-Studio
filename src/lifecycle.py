import threading


# Evento compartilhado pelas rotas long-lived e pelo servidor Uvicorn.
shutdown_requested = threading.Event()
