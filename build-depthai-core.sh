#!/bin/bash

mkdir -p build/depthai-core && \
cmake -Sdepthai-core -Bbuild/depthai-core -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
cmake --build build/depthai-core && \
cmake --install build/depthai-core --prefix build/depthai-core/install
