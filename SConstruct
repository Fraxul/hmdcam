import os
import platform
import sys
import re

vars = Variables(None, ARGUMENTS)
vars.Add(BoolVariable('debug', 'Set to build in debug mode (no optimization)', 0))

try:
  # os.cpu_count() is python3 only
  SetOption('num_jobs', os.cpu_count())
except:
  # Python2 backup strat
  import multiprocessing
  SetOption('num_jobs', multiprocessing.cpu_count())

is_tegra = (platform.machine() == 'aarch64')
tegra_release = 0

if is_tegra:
  try:
    with open('/etc/nv_tegra_release') as f:
      res = re.search('R(\d+)', f.readline())
      tegra_release = int(res.group(1))
  except:
    print("Error reading/parsing /etc/nv_tegra_release:", sys.exc_info()[0])

  if tegra_release == 0:
    print('WARNING: Unable to determine L4T release version!')

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
    CPPPATH=['/usr/include/drm', tegra_mmapi + '/include', tegra_mmapi + '/argus/include', '/usr/local/cuda/include', '/usr/local/include/opencv4', '#tegra_mmapi', '#live555/include'],
    NVCCPATH=['/usr/local/include/opencv4'],
    LIBPATH=['/usr/lib/aarch64-linux-gnu/tegra', '/usr/local/lib', '/usr/local/cuda/lib64'],
    CUDA_SDK_PATH='/usr/local/cuda',
    IS_TEGRA=True,
    TEGRA_MMAPI=tegra_mmapi,
    CPPDEFINES=[('L4T_RELEASE_MAJOR', tegra_release)]
  )

else:
  # Reduced environment for non-tegra
  env = Environment(tools = ['clang', 'clangxx', 'link', 'cuda'], toolpath=['scons-tools'],
    CPPPATH=['/usr/local/include/opencv4', '/usr/local/cuda/include'],
    NVCCPATH=['/usr/local/include/opencv4'],
    LIBPATH=['/usr/local/lib', '/usr/local/cuda/lib64'],
    CUDA_SDK_PATH='/usr/local/cuda',
    CUDA_TOOLKIT_PATH='/usr/local/cuda',
    IS_TEGRA=False
  )

vars.Update(env)

# Common env
env.Append(
  CPPPATH=['#.', '#glm', '#imgui'],
  CPPFLAGS=['-g', '-Wall'],
  CPPDEFINES=['NO_OPENSSL'],
  CXXFLAGS=['-std=c++14'],
  LINKFLAGS=['-g'],
)

if (not env['debug']):
  env.Append(CPPFLAGS=['-O2'])

have_opencv_cuda = True
conf = Configure(env)
if not conf.CheckLib('opencv_cudaimgproc'):
  print("OpenCV doesn't appear to have been built with cudaimgproc -- SHM-based backends and the RDMA client will not build.")
  have_opencv_cuda = False
conf.Finish()
if (have_opencv_cuda):
  env.Append(CPPDEFINES=['HAVE_OPENCV_CUDA'])
env['HAVE_OPENCV_CUDA'] = have_opencv_cuda


have_vpi2 = True
if not os.path.isdir('/opt/nvidia/vpi2'):
  have_vpi2 = False
  print('VPI2 library is not installed at /opt/nvidia/vpi2. VPI2 backend will not build.')
if (have_vpi2):
  env.Append(CPPDEFINES=['HAVE_VPI2'])
env['HAVE_VPI2'] = have_vpi2

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

if have_opencv_cuda:
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

