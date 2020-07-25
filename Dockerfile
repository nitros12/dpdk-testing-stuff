FROM nvidia/cuda:11.0-base-ubuntu18.04

ARG DPDK_VERSION=20.05
ARG DPDK_PATH=http://fast.dpdk.org/rel
ARG DPDK_TARGET=/usr/local/src/dpdk-${DPDK_VERSION}

RUN apt-get update \
  && apt-get install -y \
    build-essential \
    libnuma1 \
    libpcap0.8 \
    libnuma-dev \
    libpcap-dev \
    python3-setuptools \
    python3-pip \
    wget \
    iproute2 \
    pciutils \
    python \
    kmod \
    pciutils \
  && pip3 install \
    meson \
    ninja \
    wheel \
  && wget ${DPDK_PATH}/dpdk-${DPDK_VERSION}.tar.gz -O - | tar xz -C /usr/local/src

WORKDIR ${DPDK_TARGET}

RUN meson build \
  && cd build \
  && ninja \
  && ninja install \
  && ldconfig

WORKDIR /opt/test
COPY target/debug/dpdk-gpu-test dpdk-gpu-test

CMD ./dpdk-gpu-test
