IMAGE=tnsorrt
run:
	docker build -t ${IMAGE} .
in:
	NV_GPU=0 nvidia-docker run --runtime=nvidia --rm -it -v /home/hagi:/home/app ${IMAGE}:latest

