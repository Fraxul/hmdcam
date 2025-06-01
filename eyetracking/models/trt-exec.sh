#!/bin/bash
usage() {
  echo "Usage: trt-exec.sh model.engine"
  exit 1
}

if [ -z "$1" ]; then
  echo "Error: No model name specified."
  usage
fi

model=$1

#nsys profile --trace=cuda,nvtx,cublas,cudla,cusparse,cudnn --output=${model}-dla.nvvp --force-overwrite true 
/usr/src/tensorrt/bin/trtexec --loadEngine=${model} --iterations=100000 
# --idleTime=25 --duration=0 --useSpinWait --useCudaGraph

