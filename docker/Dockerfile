FROM docker.bin.sbb.ch/dskit/jupyter-base:0.23.0
LABEL author="Christian Eichenberger <u214892@sbb.ch>"
LABEL description="rsp jupyter workspace"

USER root

# install miniconda
RUN set -e && \
    set -x && \
    apt-get update && \
    apt-get -y upgrade && \
    # install dependencies for FLATland visualization: https://gitlab.aicrowd.com/flatland/flatland/blob/master/.gitlab-ci.yml
    apt-get -y install  libgl1-mesa-glx xvfb xdg-utils libcairo2-dev libjpeg-dev libgif-dev && \
    # install ffmpeg for rsp conversion
    apt-get -y install ffmpeg && \
    # install requirements for plotly/orca, see https://github.com/plotly/orca/blob/3.3-release/deployment/Dockerfile
    apt-get -y install libgtk-3-0 libgconf-2-4 libnss3 && \
    # install python3-dev because of https://github.com/giampaolo/psutil/issues/1714
    apt-get -y install --no-install-recommends gcc && \
    apt-get -y install --no-install-recommends python3-dev

RUN ls -al ~/.ssh

# setup conda env
ADD rsp_environment.yml /tmp/rsp_environment.yml
RUN conda env create -f /tmp/rsp_environment.yml

# N.B. we inherit ENTRYPOINT from dskit/jupyter-base, no need to redefine
#   see https://code.sbb.ch/projects/KD_BIGDATA/repos/dskit-jupyter-base/browse/docker/Dockerfile?at=refs%2Ftags%2F0.4.0#133-136
# N.B. we do not re-define CMD since we specify args in deployment.yaml!
# / TODO HPC-101 -> DSKIT-28  workaround for "fatal: could not create work tree dir 'hpc-quickstart-training': Permission denied"?
RUN set -e && set -x && export NB_GID=0 && fix-permissions $HOME
# Switch back to jovyan
USER $NB_USER

RUN mkdir -p $HOME/data

RUN printenv && \
    id -u && \
    id -g && \
    set-sbb-proxy && \
    python3 -m pip config set global.cert /etc/ssl/certs/ca-certificates.crt
# \  TODO HPC-101 -> DSKIT-28
