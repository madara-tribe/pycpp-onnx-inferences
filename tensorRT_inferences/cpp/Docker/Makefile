IMAGE=cpprt
run:
	docker build -t ${IMAGE} .
in:
	nvidia-docker run --runtime=nvidia --rm -it -v /home/ubuntu:/workspace ${IMAGE}:latest
