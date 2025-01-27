#!/bin/bash

model=pfld-uint8-in
nsys profile --trace=cuda,nvtx,cublas,cudla,cusparse,cudnn,nvmedia --output=${model}.nvvp /usr/src/tensorrt/bin/trtexec --loadEngine=${model}.engine --iterations=10 --idleTime=500 --duration=0 --useSpinWait --useCudaGraph

