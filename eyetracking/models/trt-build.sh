#!/bin/bash
model=pfld-uint8-in

time /usr/src/tensorrt/bin/trtexec --onnx=${model}.onnx --saveEngine=${model}.engine --exportProfile=${model}.json --timingCacheFile=timing-$(uname -i).cache --inputIOFormats=uint8:chw --outputIOFormats=fp32:chw --fp16 --allowGPUFallback --useSpinWait --separateProfileRun --useCudaGraph 2>&1 | tee ${model}.log
# --useDLACore=0

