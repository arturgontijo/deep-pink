FROM ubuntu:18.04

ENV GIT_BRANCH=py3
ENV GIT_USERNAME=erikbern
ENV GIT_REPONAME=deep-pink

# Basic tools
RUN apt update && \
    apt install -y \
    git \
    wget \
    nano

# Python3
RUN apt install -y python3 python3-pip

# Theano dependencies
RUN pip3 install --upgrade numpy==1.12.0 && \
    pip3 install git+https://github.com/Theano/Theano.git#egg=Theano

# Sunfish
RUN cd /opt && \
    git clone https://github.com/thomasahle/sunfish
ENV PYTHONPATH "${PYTHONPATH}:/opt/sunfish"

# Python Chess
RUN pip3 install python-chess

# For training
RUN pip3 install scikit-learn && \
    pip3 install h5py

RUN cd /opt && \
    git clone -b ${GIT_BRANCH} https://github.com/${GIT_USERNAME}/${GIT_REPONAME}.git

WORKDIR /opt/${GIT_REPONAME}