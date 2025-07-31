.PHONY: help run-api run-tests

help:
	@echo "Available commands:"
	@echo "run-api 		- Run FastAPI Server" 
	@echo "install-dependencies 	- Installs the dependencies" 
	@echo "run-tests 		- Runs the pytests"

run-api:
	@echo "Running FastAPI Server..."
	cd src && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

install-dependencies:
	@echo "Installing dependencies"
	cd .devcontainer/ && pip install -r requirements.txt --break-system-packages

run-tests:
	@echo "Running Test cases"
	cd tests/ && pytest 