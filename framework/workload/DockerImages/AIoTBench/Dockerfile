ARG ARCH=
FROM ${ARCH}ubuntu:18.04

WORKDIR /root

RUN \
    apt-get update && apt-get install -y \
    autoconf \
    build-essential \
    libtool \
    time \
    bc \
    python3 \
    python3-pip \
    wget

RUN pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pillow tqdm

COPY main.py .
COPY assets assets
