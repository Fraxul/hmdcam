#!/bin/bash

if [ ! -f build/shaderc/build.ninja ]; then
  # Ensure that shaderc dependencies are synced before building.
  pushd shaderc
  utils/git-sync-deps
  popd

  mkdir -p build/shaderc

  cmake -Sshaderc  -Bbuild/shaderc  -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON 
  if [ $? -ne 0 ]; then
    exit
  fi
fi

cmake --build build/shaderc && \
cmake --install build/shaderc --prefix build/shaderc/install

