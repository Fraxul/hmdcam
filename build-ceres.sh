#!/bin/bash

if [ ! -f build-ceres.sh ]; then
  echo "Must be run from the root of the repository."
  exit 1
fi

mkdir -p build
pushd build

RELEASE_PACKAGE=ceres-solver-2.2.0
RELEASE_PACKAGE_TGZ=${RELEASE_PACKAGE}.tar.gz
SRC_DIR=ceres-src
BUILD_DIR=ceres-build
OUTPUT_DIR=ceres

if [ ! -d ${SRC_DIR} ]; then
  if [ ! -f ${RELEASE_PACKAGE_TGZ} ]; then
    # Download tgz if required
    curl -L http://ceres-solver.org/${RELEASE_PACKAGE_TGZ} -o ${RELEASE_PACKAGE_TGZ}
  fi

  # Unpack tgz if required
  tar xvzf ${RELEASE_PACKAGE_TGZ}
  mv ${RELEASE_PACKAGE} ${SRC_DIR}
fi

# Create build directory
mkdir -p ${BUILD_DIR}

# Create output directory
mkdir -p ${OUTPUT_DIR}

# CMake configuration.
# Skip building examples and tests.
# We also disable CUDA because we want to avoid adding extra GPU work,
# which might disrupt the timing of the camera loop.
CC=/usr/bin/clang CXX=/usr/bin/clang++ LD=/usr/bin/lld cmake -S ${SRC_DIR} -B ${BUILD_DIR} -G Ninja -DUSE_CUDA=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$(realpath ${OUTPUT_DIR})

# build and install into output dir
ninja -C ceres-build install

# exit build dir
popd

