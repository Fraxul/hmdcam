import platform
import os

is_tegra = (platform.machine() == 'aarch64')

if is_tegra:
  # Environment setup
  env = Environment(tools = ['clang', 'clangxx', 'link', 'cuda'], toolpath=['scons-tools'],
    CPPPATH=['#tegra_mmapi', '#live555/include', '/usr/include/drm', '/usr/src/tegra_multimedia_api/include', '/usr/src/tegra_multimedia_api/argus/include', '/usr/local/cuda/include', '/usr/local/include/opencv4'],
    LIBPATH=['/usr/lib/aarch64-linux-gnu/tegra', '/usr/local/lib', '/usr/local/cuda/lib64'],
    CUDA_SDK_PATH='/usr/local/cuda',
    IS_TEGRA=True
  )

else:
  # Reduced environment for non-tegra
  # TODO CUDA path
  env = Environment(tools = ['clang', 'clangxx', 'link', 'cuda'], toolpath=['scons-tools'],
    CPPPATH=['/usr/local/include/opencv4', '/usr/local/cuda/include'],
    LIBPATH=['/usr/local/lib', '/usr/local/cuda/lib64'],
    CUDA_SDK_PATH='/usr/local/cuda',
    CUDA_TOOLKIT_PATH='/usr/local/cuda',
    IS_TEGRA=False
  )

# Common env
env.Append(
  CPPPATH=['#.', '#glm', '#glatter/include', '#glatter/include/glatter', '#imgui'],
  CPPFLAGS=['-g', '-Wall', '-O2'],
  CPPDEFINES=['NO_OPENSSL'],
  CXXFLAGS=['-std=c++11'],
  LINKFLAGS=['-g'],
)

# Fix for clang colored diagnostics
env['ENV']['TERM'] = os.environ['TERM']
env.Decider('MD5-timestamp')
Export('env')

if is_tegra:
  # Only build hmdcam application on Tegra
  SConscript('SConscript-hmdcam', variant_dir = 'build/hmdcam', duplicate = 0)

SConscript('SConscript-rdma-client', variant_dir = 'build/rdma-client', duplicate = 0)
SConscript('SConscript-dgpu-worker', variant_dir = 'build/dgpu-worker', duplicate = 0)

