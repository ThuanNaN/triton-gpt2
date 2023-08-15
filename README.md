# Deploy GPT2 model to triton server

## Repare image
```bash
git clone https://github.com/ThuanNaN/triton-gpt2.git
cd triton-gpt2
docker compose up
```

## Start triton server
### Start workspace (terminal 1)
```bash
# add flag --gpus=all if run container with GPU
docker run --shm-size=256m -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models triton_img
```
### Start tritron server
```bash
tritonserver --model-repository=/models
```


## Inference 
### Start workspace (terminal 2)
```bash
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.07-py3-sdk bash
```

### Test model
```bash
python client.py --prompt "Say somethings"
```

down_all:
	make triton_down
	make fastapi_down

up_all:
	make triton_up
	make fastapi_up

triton_up:
	PWD=$(pwd) docker compose -f ./docker/tritonserver/docker-compose.yml up -d

triton_down:
	PWD=$(pwd) docker compose -f ./docker/tritonserver/docker-compose.yml down

fastapi_up:
	PWD=$(pwd) docker compose -f ./docker/fastapi/docker-compose.yml up -d

fastapi_down:
	PWD=$(pwd) docker compose -f ./docker/fastapi/docker-compose.yml down