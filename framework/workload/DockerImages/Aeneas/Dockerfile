FROM ubuntu:18.04


WORKDIR /root

RUN \
    apt-get update && apt-get install -y \
    autoconf \
    build-essential \
    libtool \
    time \
    bc \
    python \
    python-pip \
    git

RUN \
    apt-get install -y \
    wget 
    
RUN \
    pip install boto3   

RUN \
    git init && \
    git remote add -f origin https://github.com/qub-blesson/DeFog.git && \
    git config core.sparsecheckout true && \
    echo Aeneas/aeneas/ >> .git/info/sparse-checkout && \
    git pull https://github.com/qub-blesson/DeFog.git master
    
RUN \
    cd Aeneas/aeneas && \
    bash install_dependencies.sh && \
    pip install -r requirements.txt && \
    python setup.py build_ext --inplace
    
COPY execute.sh .
RUN chmod +x execute.sh

COPY assets assets

CMD ["./execute.sh"]
