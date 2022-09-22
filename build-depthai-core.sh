#!/bin/bash

if [ ! -d build/depthai-core ]; then
  mkdir -p build/depthai-core
  # Sample for building with local XLink: -DDEPTHAI_XLINK_LOCAL=/home/hmdcam/XLink
  XLINK_REPO=`pwd`/XLink
  if [ ! -d $XLINK_REPO ]; then
    echo "XLink local repository couldn't be found at ${XLINK_REPO}. Make sure submodules are up to date: git submodule update --init --recursive"
    exit 1
  fi
  cmake -Sdepthai-core -Bbuild/depthai-core -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_FLAGS="-fno-omit-frame-pointer -funwind-tables" -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -funwind-tables" -DDEPTHAI_XLINK_LOCAL=${XLINK_REPO}
  if [ $? -ne 0 ]; then
    exit
  fi
fi

cmake --build build/depthai-core && \
cmake --install build/depthai-core --prefix build/depthai-core/install

