.PHONY: help run-api 

help:
	@echo "Available commands:"
	@echo "run-api - Run FastAPI Server" 

run-api:
	@echo "Running FastAPI Server..."
	cd src && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

install-dependencies:
	