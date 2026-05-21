sudo docker run --rm -it \
    --gpus all \
    --network host \
    --ipc host \
    --name test \
    --privileged \
    -v /host_dir:/container_dir \
    image-repository:image-tag
