FROM getcapsule/sandbox:19.11.1-1.43 as source

ARG DPDK_VERSION=19.11.1
ARG DPDK_TARGET=/usr/local/src/dpdk-stable-${DPDK_VERSION}

FROM nvidia/cuda:11.0-base-ubuntu18.04

COPY --from=source /usr/local/bin /usr/local/bin
COPY --from=source /usr/local/lib/x86_64-linux-gnu /usr/local/lib/x86_64-linux-gnu
COPY --from=source /usr/local/include /usr/local/include
COPY --from=source ${DPDK_TARGET} ${DPDK_TARGET}

WORKDIR /opt/test
COPY target/debug/dpdk-gpu-test dpdk-gpu-test

CMD ./dpdk-gpu-test
