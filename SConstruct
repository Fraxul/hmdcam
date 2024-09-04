import os
import platform
import sys
import re
import SCons

vars = Variables(None, ARGUMENTS)
vars.Add(BoolVariable('debug', 'Set to build in debug mode (no optimization)', 0))
vars.Add(BoolVariable('nsight', 'Set to build for NSight compatibility (disables NVEnc, VPI2)', 0))
vars.Add(BoolVariable('cuda_debug', 'Set to build CUDA kernels in debug mode', 0))


env_tools = ['clang', 'clangxx', 'link', 'cuda']

scons_version_major = int(SCons.__version__.split('.')[0])
if (scons_version_major >= 4):
  # compilation_db support was added in SCons 4.0
  env_tools += ['compilation_db']

try:
  # os.cpu_count() is python3 only
  SetOption('num_jobs', os.cpu_count())
except:
  # Python2 backup strat
  import multiprocessing
  SetOption('num_jobs', multiprocessing.cpu_count())


# Base environment setup
env = Environment(tools = env_tools, toolpath=['scons-tools'],
  CPPPATH=[
    '#.',
    '#glm',
    '#imgui',
    '/usr/local/cuda/include',
    '/usr/local/include/opencv4',
  ],
  NVCCPATH=[
    '.',
    '/usr/local/include/opencv4',
    'glm'
  ],
  NVCCFLAGS=['--expt-relaxed-constexpr', '-g'],
  CPPDEFINES=['GLM_ENABLE_EXPERIMENTAL'],
  CPPFLAGS=['-g', '-Wall'],
  CXXFLAGS=['-std=c++14'],
  LINKFLAGS=['-g'],
  LIBPATH=['/usr/lib/aarch64-linux-gnu/tegra', '/usr/local/lib', '/usr/local/cuda/lib64'],
  CUDA_SDK_PATH='/usr/local/cuda',
  COMPILATIONDB_USE_ABSPATH=True
)

vars.Update(env)

if (env['cuda_debug']):
  # Compile device code in debug mode -- no optimizations
  env.Append(NVCCFLAGS=['-G'])
else:
  # Generate debug line number info for device code
  env.Append(NVCCFLAGS=['-lineinfo'])

if (scons_version_major < 4):
  def NullCompilationDatabase():
    pass

  env.CompilationDatabase = NullCompilationDatabase

# Fix for clang colored diagnostics
env['ENV']['TERM'] = os.environ['TERM']
env.Decider('MD5-timestamp')

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

  # TODO: Correctly detect the CUDA codegen version
  # Compute capability 8.7 works for Orin parts
  env.Append(NVCCFLAGS=['--generate-code', 'arch=compute_87,code=sm_87'])

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
  env['TEGRA_MMAPI'] = tegra_mmapi
  env['IS_TEGRA'] = True
  env['TEGRA_RELEASE'] = tegra_release
  env.Append(
    CPPDEFINES=[('L4T_RELEASE_MAJOR', tegra_release)]
  )

else:
  # Reduced environment for non-tegra
  env['IS_TEGRA'] = False
  if (platform.platform().find('WSL2') >= 0):
    env.Append(LIBPATH=['/usr/lib/wsl/lib'])

# Common env
if (env['debug']):
  env.Append(NVCCFLAGS=['--debug', '--device-debug'])
else:
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


have_vpi2 = False
if (not env['nsight']):
  if os.path.isdir('/opt/nvidia/vpi2'):
    have_vpi2 = True
    env.Append(CPPDEFINES=['HAVE_VPI2'])
  else:
    print('VPI2 library is not installed at /opt/nvidia/vpi2. VPI2 backend will not build.')

env['HAVE_VPI2'] = have_vpi2

Export('env')

build_dgpu = True
if (is_tegra and (not os.path.isdir('/usr/local/nvidia-dgpu-support'))):
  build_dgpu = False
  print('DGPU support libraries for Tegra are not installed at /usr/local/nvidia-dgpu-support. DGPU backend will not build.')

SConscript('SConscript-hmdcam', variant_dir = 'build/hmdcam', duplicate = 0)
SConscript('SConscript-pdu-test', variant_dir = 'build/pdu-test', duplicate = 0)
if (is_tegra and build_dgpu):
  SConscript('SConscript-pductl', variant_dir = 'build/pductl', duplicate = 0)
  SConscript('SConscript-dgpu-fans', variant_dir = 'build/dgpu-fans', duplicate = 0)

if not is_tegra:
  # Only build test apps on desktop
  SConscript('SConscript-stereo-geometry', variant_dir = 'build/stereo-geometry', duplicate = 0)

if have_opencv_cuda:
  SConscript('SConscript-rdma-client', variant_dir = 'build/rdma-client', duplicate = 0)
SConscript('SConscript-eyectl', variant_dir = 'build/eyectl', duplicate = 0)
if (build_dgpu):
  SConscript('SConscript-dgpu-worker', variant_dir = 'build/dgpu-worker', duplicate = 0)

# SHM worker benchmark framework
SConscript('SConscript-worker-benchmark', variant_dir = 'build/worker-benchmark', duplicate = 0)

# DepthAI worker (optional)
if os.path.isdir('build/depthai-core/install'):
  SConscript('SConscript-depthai-worker', variant_dir = 'build/depthai-worker', duplicate = 0)
else:
  print('Did not find build artifacts from depthai-core -- the DepthAI worker will be disabled.')
  print('If you want to build the DepthAI worker, run ./build-depthai-core.sh first')

