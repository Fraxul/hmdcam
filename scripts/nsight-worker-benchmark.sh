#!/bin/bash
QUADD_INJECTION_PROXY="OSRT, $QUADD_INJECTION_PROXY" LD_PRELOAD="/opt/nvidia/nsight_systems/libToolsInjectionProxy64.so" NVTX_INJECTION64_PATH="/opt/nvidia/nsight_systems/libToolsInjection64.so" build/bin/worker-benchmark --depth-backend depthai --res 480 270 --views 2

