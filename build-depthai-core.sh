#!/bin/bash

if [ ! -d build/depthai-core ]; then
  mkdir -p build/depthai-core
  # Sample for building with local XLink: -DDEPTHAI_XLINK_LOCAL=/home/hmdcam/XLink
  cmake -Sdepthai-core -Bbuild/depthai-core -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
  if [ $? -ne 0 ]; then
    exit
  fi
fi

cmake --build build/depthai-core && \
cmake --install build/depthai-core --prefix build/depthai-core/install

