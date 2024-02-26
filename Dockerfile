FROM haosulab/mani-skill2:latest
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENV YOUR_CODES_AND_WEIGHTS_IN_HOST .
ENV YOUR_CODES_AND_WEIGHTS_IN_CONTAINER /root/

# Upgrade conda
ENV PATH /opt/conda/lib:$PATH 
RUN conda install -n base -c defaults conda=24.1.0

RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -y
# CUDAToolkit is a prerequisite for CUDA
RUN conda install cuda -c nvidia/label/cuda-11.3.1 -y

# Copy your codes (including user_solution.py) and model weights
COPY ${YOUR_CODES_AND_WEIGHTS_IN_HOST} ${YOUR_CODES_AND_WEIGHTS_IN_CONTAINER}
ENV PYTHONPATH ${YOUR_CODES_AND_WEIGHTS_IN_CONTAINER}:${PYTHONPATH}

# Install additional python packages you need
RUN pip install --upgrade pip 
RUN pip install -r ${YOUR_CODES_AND_WEIGHTS_IN_CONTAINER}requirements.txt 
RUN pip cache purge

# RUN apt-get update
# # Install TensorRT
# # ENV os "ubuntu2004"
# # ENV tag "8.6.1-cuda-11.8"
# # RUN dpkg -i ${YOUR_CODES_AND_WEIGHTS_IN_CONTAINER}Downloads/nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
# # RUN cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
# # RUN apt-get install python3-libnvinfer-dev -y

# # Install CUDA
# # RUN dpkg -i {YOUR_CODES_AND_WEIGHTS_IN_CONTAINER}Downloads/cuda-keyring_1.1-1_all.deb
# # RUN apt-get -y install cuda-toolkit-12-3

# # RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so.8 /usr/lib/x86_64-linux-gnu/libnvinfer.so.7
# # RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

# # Ensure that all Vulkan extensions are installed
# RUN apt-get upgrade -y --no-install-recommends\
#     bash-completion \
#     build-essential \
#     ca-certificates \
#     cmake \
#     htop \
#     libegl1 \
#     libxext6 \
#     libjpeg-dev \
#     libpng-dev  \
#     libvulkan1
