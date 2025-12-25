.PHONY: start-backend start-frontend

start-backend:
	docker compose up -d && uv run uvicorn src.backend.app.main:app --reload --host 0.0.0.0 --port 8000

start-frontend:
	cd src/frontend && npm run dev