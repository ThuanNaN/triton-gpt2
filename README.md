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

