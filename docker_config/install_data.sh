#! /usr/bin/zsh

if [ $GPU_CONTAINER ]
then
    CONTAINER_TAG=jamie/nlp_environment_gpu
else
    CONTAINER_TAG=jamie/nlp_environment_cpu
fi


docker run \
    -u $(id -u):$(id -g) \
    --name tensorflow_gpu_inst \
    --mount type=bind,src="$(pwd)"/app,target=/app \
    --gpus all \
    -it \
    --env-file env_file.txt \
    $CONTAINER_TAG \
    python3 /app/src/setup_nltk_data.py