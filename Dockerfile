FROM tensorflow/tensorflow:latest-py3-jupyter as base

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    # && apt-get install -y --no-install-recommends \
    && apt-get install -y \
    rsync \
    gzip \
    bzip2 \
    xz-utils \
    liblzma5 \
    liblzma-dev \
    zlib1g \
    less \
    libbz2-dev \
    libzstd1-dev \
    libcdk5-dev \
    libarchive-dev \
    libz-dev \
    libstarlink-pal-dev \
    libopenblas-dev \
    python3-dev \
    python3-scipy \
    python3-numpy \
    python3-numexpr \
    python3-urwid \
    python3-cffi \
    python3-healpy \
    python3-urllib3 \
    python3-jsonschema \
    python3-requests \
    cython3 \
    python3-zmq \
    python3-pymysql \
    build-essential \
    cmake \
    zlib1g-dev \
    libboost-all-dev \
    zstd \
    libgmp3-dev \
    libxml2-dev \
    libfftw3-dev \
    libsprng2-dev \
    libgsl-dev \
    libsuitesparse-dev \
    libncurses-dev \
    libncursesw5-dev \
    libcdk5-dev \
    libcfitsio-dev \
    libhealpix-cxx-dev \
    libhdf5-serial-dev \
    libclhep-dev \
    opencl-headers \
    opencl-c-headers \
    opencl-clhpp-headers \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    liblog4cpp5-dev \
    libzmq5 \
    libzmq3-dev \
    libzmqpp-dev \
    libzmqpp4 \
    libnlopt-dev \
    libzstd-dev \
    libblosc-dev \
    subversion \
    wget \
    vim \
    fish \
    npm \
    nodejs \
    && update-alternatives --install /usr/bin/python python \
    /usr/bin/python3 0 \
    # && mkdir /opt/i3-data /opt/i3-data/i3-test-data \
    && useradd -ms /bin/bash icecube

WORKDIR /home/icecube

USER icecube

RUN mkdir /home/icecube/combo && mkdir /home/icecube/combo/build \
    && svn co http://code.icecube.wisc.edu/svn/meta-projects/combo/stable \
    /home/icecube/combo/src --username=icecube --password=skua \
    --no-auth-cache

WORKDIR /home/icecube/combo/build

RUN cmake /home/icecube/combo/src \
    && make -j2

FROM base as extras

USER root

run apt-get install -y \
    python3-pip \
    python3-setuptools \
    python3-wheel

USER icecube

ENV PATH="/home/icecube/.local/bin:${PATH}"

RUN pip3 install --user streamlit plotly pandas tables scikit-learn

ENTRYPOINT ["/bin/bash", "/home/icecube/combo/build/env-shell.sh", "exec"]
CMD ["/bin/bash"]
