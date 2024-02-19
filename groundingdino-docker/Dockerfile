FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

COPY sources_bionic.list /etc/apt/sources.list
ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

RUN apt update && \
     apt install openjdk-17-jdk -y

RUN apt install git -y

#install python packages
COPY . /ws
RUN pip install -r /ws/requirements.txt --no-cache -i https://repo.huaweicloud.com/repository/pypi/simple/
RUN cd /ws/GroundingDINO/ && python -m pip install -e . 