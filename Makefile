export USER_ID      := $(shell id -u)
export GROUP_ID     := $(shell id -g)
export USER_NAME    := $(shell whoami)
export PROJECT_DIR  := $(shell pwd)
export PROJECT_NAME := pytorch-interactive-${USER_NAME}

SHELL := /bin/bash

export COMPOSE_CMD  := source .env && docker-compose -f docker/docker-compose.yml -p ${PROJECT_NAME}

build:
	$(COMPOSE_CMD) build

run:
	${COMPOSE_CMD} run interactive bash
