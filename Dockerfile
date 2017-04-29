FROM ubuntu:16.04

# See for instance https://github.com/janelia-flyem/flyem-dockerfiles/tree/master/dvid

MAINTAINER Carsten Haubold <carsten.haubold@iwr.uni-heidelberg.de>
# goes to https://hub.docker.com/r/hcichaubold/ilastikbackend/

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda create -n ilastikenv python=3.5
ENV PATH /opt/conda/envs/ilastikenv/bin:$PATH
ENV CONDA_PREFIX /opt/conda/envs/ilastikenv
ENV CONDA_DEFAULT_ENV ilastikenv
RUN conda install -y flask redis-py requests && \
    pip install Flask_Autodoc && \
    pip install pika
RUN conda install -y ilastikbackend h5py numpy -c chaubold -c conda-forge
RUN conda install -y fastfilters -c chaubold -c conda-forge

# free at least some space, conda is eating up a lot of memory!
RUN conda clean --all -y

# TODO: build LINUX version of those packages!
COPY ./services/ /var/ilastik/services/

EXPOSE 8888 8889 8080 9000
WORKDIR /var/ilastik/services/
# CMD ["python", "service.py", "--port", "8888", "--project", "../test/pc.ilp", "--raw-data-file", "../test/raw.h5", "--raw-data-path", "exported_data", "--use-caching"]

# lastly, run that container by specifying a local volume? See
# https://docs.docker.com/engine/tutorials/dockervolumes/#mount-a-host-directory-as-a-data-volume