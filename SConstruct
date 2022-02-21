import platform
import os

is_tegra = (platform.machine() == 'aarch64')

if is_tegra:
  tegra_mmapi_paths = [
    '/usr/src/tegra_multimedia_api',
    '/usr/src/jetson_multimedia_api'
  ]
  tegra_mmapi = None
  for path in tegra_mmapi_paths:
    if (os.path.isdir(path)):
      tegra_mmapi = path
      break
  if tegra_mmapi is None:
    sys.exit('Cannot find the Tegra Multimedia API')

  # Environment setup
  env = Environment(tools = ['clang', 'clangxx', 'link', 'cuda'], toolpath=['scons-tools'],
    CPPPATH=['#tegra_mmapi', '#live555/include', '/usr/include/drm', tegra_mmapi + '/include', tegra_mmapi + '/argus/include', '/usr/local/cuda/include', '/usr/local/include/opencv4'],
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
  CPPPATH=['#.', '#glm', '#imgui'],
  CPPFLAGS=['-g', '-Wall', '-O2'],
  CPPDEFINES=['NO_OPENSSL'],
  CXXFLAGS=['-std=c++11'],
  LINKFLAGS=['-g'],
)

# Fix for clang colored diagnostics
env['ENV']['TERM'] = os.environ['TERM']
env.Decider('MD5-timestamp')
Export('env')

build_dgpu = True
if (is_tegra and (not os.path.isdir('/usr/local/nvidia-dgpu-support'))):
  build_dgpu = False
  print('DGPU support libraries for Tegra are not installed at /usr/local/nvidia-dgpu-support. DGPU backend will not build.')

if is_tegra:
  # Only build hmdcam application on Tegra
  SConscript('SConscript-hmdcam', variant_dir = 'build/hmdcam', duplicate = 0)
  if (build_dgpu):
    SConscript('SConscript-pductl', variant_dir = 'build/pductl', duplicate = 0)
    SConscript('SConscript-dgpu-fans', variant_dir = 'build/dgpu-fans', duplicate = 0)
else:
  # Only build test apps on desktop
  SConscript('SConscript-stereo-geometry', variant_dir = 'build/stereo-geometry', duplicate = 0)

SConscript('SConscript-rdma-client', variant_dir = 'build/rdma-client', duplicate = 0)
SConscript('SConscript-eyectl', variant_dir = 'build/eyectl', duplicate = 0)
if (build_dgpu):
  SConscript('SConscript-dgpu-worker', variant_dir = 'build/dgpu-worker', duplicate = 0)


# DepthAI worker (optional)
if os.path.isdir('build/depthai-core/install'):
  SConscript('SConscript-depthai-worker', variant_dir = 'build/depthai-worker', duplicate = 0)
else:
  print('Did not find build artifacts from depthai-core -- the DepthAI worker will be disabled.')
  print('If you want to build the DepthAI worker, run ./build-depthai-core.sh first')

