DOCKER_BUILDKIT=1 sudo docker build -t image-repository:image-tag-20250704 -f docker/Dockerfile_Hopper . --progress=plain 2>&1 | tee docker_build.log
