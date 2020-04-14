FROM docker.bin.sbb.ch/dskit/jupyter-base:0.23.0
LABEL maintainer="Data Science Kit Developers <xdsk001@sbb.ch>"
LABEL author="Christian Eichenberger <u214892@sbb.ch>"
LABEL description="hpc quickstart jupyter workspace"

# install miniconda
RUN set -e && \
    set -x && \
    echo $PWD && \
    mkdir -p conda_dir && \
    wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p $PWD/conda_dir && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh && \
    export PATH=$PWD/conda_dir/bin:$PATH && \
    which python && \
    which conda && \
    conda update -n root conda -y && \
    conda install -c conda-forge tox-conda

# setup conda env
ADD ../rsp_environment.yml /tmp/rsp_environment.yml
RUN conda install -f /tmp/rsp_environment.yml

# N.B. we inherit ENTRYPOINT from dskit/jupyter-base, no need to redefine
#   see https://code.sbb.ch/projects/KD_BIGDATA/repos/dskit-jupyter-base/browse/docker/Dockerfile?at=refs%2Ftags%2F0.4.0#133-136
# N.B. we do not re-define CMD since we specify args in deployment.yaml!
# / TODO HPC-74 -> DSKIT-28  workaround for "fatal: could not create work tree dir 'hpc-quickstart-training': Permission denied"?
USER root
RUN set -e && set -x && export NB_GID=0 && fix-permissions $HOME
# Switch back to jovyan
USER $NB_USER
# \  TODO HPC-74 -> DSKIT-28
