# AI Training Studio

Ambiente local para treinar, adaptar, avaliar e executar modelos de inteligência artificial.

## Execução local

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

## Docker

```bash
docker compose up -d --no-build app
```
