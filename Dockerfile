FROM nvcr.io/nvidia/tensorrt:22.12-py3

SHELL ["/bin/bash", "-c"]

# Setup user account
# id -g, id -u
ARG uid=1008
ARG gid=1008
RUN groupadd -r -f -g ${gid} hpaik && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash hpaik
RUN usermod -aG sudo hpaik
RUN echo 'hpaik:hpaik1' | chpasswd
RUN mkdir -p /workspace && chown hpaik /workspace

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# locale Korean
# RUN apt-get update && apt-get install -y locales sudo libcholmod3
# RUN locale-gen ko_KR.UTF-8
# ENV LC_ALL ko_KR.UTF-8

# install
RUN apt-get update && apt-get install -y sudo && \
    apt-get install -y libgl1-mesa-glx git locales && \
    locale-gen ko_KR.UTF-8
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /workspace

USER hpaik
RUN ["/bin/bash"]