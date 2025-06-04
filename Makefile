#!/bin/sh
.PHONY: build dev down ssh publish
build:
	docker image rm -f izdrail/images.laravelcompany.com:latest && docker build --no-cache -t izdrail/images.laravelcompany.com:latest --progress=plain .
	docker-compose -f docker-compose.yml up  --remove-orphans

dev:
	docker-compose up

down:
	docker-compose down
ssh:
	docker exec -it images.laravelcompany.com /bin/zsh
publish:
	docker push izdrail/images.laravelcompany.com:latest
