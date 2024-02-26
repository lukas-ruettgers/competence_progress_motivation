export DOCKER_USER=ltgslks23
export DOCKER_IMAGE=mani-skill2-submission
export IMAGE_TAG=cpm010

# Build the image.
cd /home/lukasr/EmbAI/Submission/
docker build . -t ${DOCKER_IMAGE} --network host
docker tag ${DOCKER_IMAGE} ${DOCKER_USER}/${DOCKER_IMAGE}:${IMAGE_TAG}
docker push ${DOCKER_USER}/${DOCKER_IMAGE}:${IMAGE_TAG}
