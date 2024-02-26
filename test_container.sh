export DOCKER_IMAGE=mani-skill2-submission
export CONTAINER_NAME=mani-skill2-evaluation
# Ensure no conflicting container is already running.
docker rm ${CONTAINER_NAME}

# Build the image.
cd /home/lucas/EmbAI/EmbAI-Final-Project/Submission
docker build . -t ${DOCKER_IMAGE} --network host

# Initialize a detached container.
docker run -d -t --name ${CONTAINER_NAME} --network host ${DOCKER_IMAGE}:latest

# Run the evaluation.
docker exec -it ${CONTAINER_NAME} /bin/bash -c \
"export MS2_ASSET_DIR=/root/data; python -m mani_skill2.evaluation.run_evaluation -e PickCube-v0 -o /eval_results/PickCube-v0 -n 1"

# Delete the container.
docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

# Remove the image.
docker rmi ${DOCKER_IMAGE}:latest
