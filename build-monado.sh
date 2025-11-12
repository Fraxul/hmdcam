#!/bin/bash

# Generate build directory if it doesn't already exist
if [ ! \( -f build/monado/build.ninja -a -f build/monado/CMakeFiles/rules.ninja \) ]; then
  mkdir -p build/monado && \
  cmake -Smonado -Bbuild/monado \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_C_FLAGS=-Wno-strict-prototypes \
    -DCMAKE_BUILD_TYPE=Debug \
    -DXRT_FEATURE_SERVICE=OFF \
    -DXRT_FEATURE_COMPOSITOR_MAIN=OFF \
    -DXRT_FEATURE_OPENXR=OFF \
    -DXRT_HAVE_WAYLAND=OFF \
    -DXRT_HAVE_XLIB=OFF \
    -DXRT_HAVE_XRANDR=OFF \
    -DXRT_HAVE_XCB=OFF \
    -DXRT_HAVE_OPENGL=OFF \
    -DXRT_HAVE_OPENGL_GLX=OFF \
    -DXRT_HAVE_OPENGLES=OFF \
    -DXRT_HAVE_VULKAN=OFF \
    -DXRT_HAVE_EGL=OFF \
    -DXRT_HAVE_SDL2=OFF \
    -DXRT_HAVE_DBUS=OFF \
    -DXRT_HAVE_JPEG=OFF \
    -DXRT_HAVE_LIBUVC=OFF \
    -DXRT_HAVE_OPENCV=OFF \
    -DXRT_HAVE_FFMPEG=OFF \
    -DXRT_FEATURE_STEAMVR_PLUGIN=OFF \
    -DXRT_BUILD_DRIVER_HANDTRACKING=OFF \
    -DXRT_BUILD_DRIVER_VF=OFF \
    -DBUILD_TESTING=OFF \
    -G Ninja
  # Ensure cmake finished before trying to build
  if [ $? -ne 0 ]; then
    exit
  fi
fi

# Build existing build directory
cmake --build build/monado

