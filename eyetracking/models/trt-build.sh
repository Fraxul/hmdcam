#!/bin/bash

usage() {
  echo "Usage: trt-build.sh model.onnx {gpu|dla}"
  exit 1
}

if [ -z "$1" ]; then
  echo "Error: No model name specified."
  usage
fi


if [ "$2" == "gpu" ]; then
  target_args="--useCudaGraph --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw"
  
elif [ "$2" == "dla" ]; then
  target_args="--useDLACore=0 --allowGPUFallback --directIO --inputIOFormats=fp16:dla_linear --outputIOFormats=fp16:dla_linear" # --memPoolSize=dlaSRAM:1"

elif [ "$2" == "dla-standalone" ]; then
  target_args="--useDLACore=0 --buildDLAStandalone --directIO --inputIOFormats=fp16:dla_linear --outputIOFormats=fp16:dla_linear" # --memPoolSize=dlaSRAM:1"

elif [ -z "$2" ]; then
  echo "Missing target"
  usage
else
  echo "Unrecognized target $2"
  usage
fi

model=$(basename "$1" .onnx)
target=$2
echo "Model: ${model}"
echo "Target: $2"


time /usr/src/tensorrt/bin/trtexec --onnx="${model}.onnx" --saveEngine="${model}-${target}.engine" --exportProfile="${model}-${target}.json" --timingCacheFile=timing-$(uname -i).cache --fp16 ${target_args} --useSpinWait --separateProfileRun --dumpProfile --dumpLayerInfo --verbose 2>&1 | tee ${model}-${target}.log
# --useDLACore=0 --allowGPUFallback --inputIOFormats=fp16:chw --profilingVerbosity=detailed 

