.PHONY: help build up down restart logs ps shell clean run-local

APP_NAME := smart-payment-router-backend
SERVICE  := api
PORT     := 8000

help:
	@echo "Targets:"
	@echo "  make build       Build the Docker image"
	@echo "  make up          Start the stack (docker compose up -d)"
	@echo "  make down        Stop the stack"
	@echo "  make restart     Restart the stack"
	@echo "  make logs        Tail logs"
	@echo "  make ps          Show running containers"
	@echo "  make shell       Open a shell inside the running container"
	@echo "  make clean       Remove containers and local __pycache__"
	@echo "  make run-local   Run locally with uvicorn (no Docker)"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose down
	docker compose up -d

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

shell:
	docker compose exec $(SERVICE) /bin/bash || docker compose exec $(SERVICE) /bin/sh

clean:
	docker compose down --remove-orphans
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

run-local:
	uvicorn main:app --host 0.0.0.0 --port $(PORT)

