.PHONY: help

IMAGE_NAME ?= "nitros12/dpdk-gpu-test"

help:
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build:
	docker build -t $(IMAGE_NAME):latest .

run: build
	docker run --rm --gpus all --privileged --net=host -v /lib/modules:/lib/modules $(IMAGE_NAME):latest
