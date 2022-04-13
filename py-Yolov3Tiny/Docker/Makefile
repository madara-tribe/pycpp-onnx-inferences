IMAGE=onnx-keras
run:
	docker build -t ${IMAGE} .
in:
	nvidia-docker run --runtime=nvidia -v /home/hagi:/root/sig --rm -it ${IMAGE}:latest
