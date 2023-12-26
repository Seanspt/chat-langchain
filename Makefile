.PHONY: start
start:
	uvicorn main:app --reload --port 8080 --host 0.0.0.0

.PHONY: format
format:
	black .
	isort .
