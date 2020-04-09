FROM docker.bin.sbb.ch/dskit/jupyter-base:0.23.0
LABEL maintainer="Data Science Kit Developers <xdsk001@sbb.ch>"
LABEL author="Christian Eichenberger <u214892@sbb.ch>"
LABEL description="hpc quickstart jupyter workspace"

# install requirements
ADD requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# N.B. we inherit ENTRYPOINT from dskit/jupyter-base, no need to redefine
#   see https://code.sbb.ch/projects/KD_BIGDATA/repos/dskit-jupyter-base/browse/docker/Dockerfile?at=refs%2Ftags%2F0.4.0#133-136
# N.B. we do not re-define CMD since we specify args in deployment.yaml!


# / TODO HPC-74 -> DSKIT-28  workaround for "fatal: could not create work tree dir 'hpc-quickstart-training': Permission denied"?
# + whoami
# ubuntu
# + id -u
# 1000810000
# + id -g
# 0

USER root
RUN set -e && set -x && export NB_GID=0 && fix-permissions $HOME
# Switch back to jovyan
USER $NB_USER
# \  TODO HPC-74 -> DSKIT-28

# TODO SIM-194 install conda environment and rsp
