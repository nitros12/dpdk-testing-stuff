# Playing around with dpdk and stuff

## setup

- `echo 256 | sudo tee
  /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages`
- `sudo env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"
  target/debug/dpdk-gpu-test`
