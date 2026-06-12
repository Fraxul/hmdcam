import os
import platform
import sys
import re
import SCons
import SCons.Util

vars = Variables(None, ARGUMENTS)
vars.Add(BoolVariable('debug', 'Set to build in debug mode (no optimization)', 0))
vars.Add(BoolVariable('nsight', 'Set to build for NSight compatibility (disables NVEnc)', 0))
vars.Add(BoolVariable('cuda_debug', 'Set to build CUDA kernels in debug mode', 0))


env_tools = ['clang', 'clangxx', 'link', 'cuda', 'ar']

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
    '#vulkan',
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
  CPPFLAGS=['-g', '-Wall', '-Wshadow'],
  CXXFLAGS=['-std=c++17'],
  LINKFLAGS=['-g', '-fuse-ld=lld'],
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

# Wrap one or more static library Nodes (results of env.StaticLibrary()) with
# linker --whole-archive / --no-whole-archive so every object gets pulled in,
# not just those that resolve an undefined symbol. Use this for libraries that
# rely on static-initializer side effects (self-registering constructors).
#
# WholeArchive() returns a list of LINKFLAGS contributions; the caller is
# responsible for also registering Depends() so SCons rebuilds the program
# when the library changes. ProgramWithWholeArchive() does both in one call:
#     env.ProgramWithWholeArchive(target=..., source=[...],
#                                 whole_archive_libs=[librhi, libimgui])
def WholeArchive(env, libs):
  libs = SCons.Util.flatten([libs])
  paths = [n.abspath for n in libs]
  return [env.Literal('-Wl,--whole-archive')] + paths + [env.Literal('-Wl,--no-whole-archive')]

def ProgramWithWholeArchive(env, target, source, whole_archive_libs, **kw):
  libs = SCons.Util.flatten([whole_archive_libs])
  kw['LINKFLAGS'] = kw.get('LINKFLAGS', env['LINKFLAGS']) + env.WholeArchive(libs)
  prog = env.Program(target=target, source=source, **kw)
  env.Depends(prog, libs)
  return prog

env.AddMethod(WholeArchive, 'WholeArchive')
env.AddMethod(ProgramWithWholeArchive, 'ProgramWithWholeArchive')

is_tegra = (platform.machine() == 'aarch64')
tegra_release = 0

if is_tegra:
  try:
    with open('/etc/nv_tegra_release') as f:
      res = re.search(r'R(\d+)', f.readline())
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
    CPPDEFINES=[('L4T_RELEASE_MAJOR', tegra_release), 'IS_TEGRA'],
    CXXFLAGS=['-march=armv8.2-a+fp16'],
  )

else:
  # Reduced environment for non-tegra
  env['IS_TEGRA'] = False
  if (platform.platform().find('WSL2') >= 0):
    env.Append(LIBPATH=['/usr/lib/wsl/lib'])

# Common env
if (env['debug']):
  env.Append(NVCCFLAGS=['--debug', '--device-debug'])
  env.Append(CPPFLAGS=['-fstandalone-debug'])
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

# Add locally-built Ceres solver
CERES_BUILD='#build/ceres'
if not os.path.exists(str(Dir(CERES_BUILD))):
  sys.exit('Ceres Solver build artifacts are not present. Please run ./build-ceres.sh first.')
env.Append(
  CPPPATH=[CERES_BUILD + '/include'],
  LIBPATH=[CERES_BUILD + '/lib']
)
env['CERES_LIBS'] = ['ceres', 'cholmod', 'lapack', 'spqr', 'glog', 'gflags']

# Sanity-check for shaderc build (required for RHI)
SHADERC_BUILD='#build/shaderc'
if not os.path.exists(str(Dir(SHADERC_BUILD))):
  sys.exit('shaderc build artifacts are not present. Please run ./build-shaderc.sh first.')
env.Append(
  CPPPATH=[SHADERC_BUILD + '/install/include'],
  LIBPATH=[SHADERC_BUILD + '/install/lib']
)

# Finally, export environment for individual component build scripts to clone and modify.
Export('env')

##### Build common libraries #####

# RHI library. This has no config dependencies besides <opencv2/config.h> (for rhi/cuda/RHICVInterop)
# Map RHI source tree to the build location
env.VariantDir('build/librhi', 'rhi', duplicate=False)
# source inputs are from the VariantDir mapping, otherwise the object files end up in the source tree.
librhi = env.StaticLibrary(
  target = '#build/lib/rhi',
  source = Glob('build/librhi/*.cpp') +
    Glob('build/librhi/cuda/*.cpp') +
    Glob('build/librhi/gl/*.cpp') +
    Glob('build/librhi/egl/*.cpp') +
    Glob('build/librhi/imgui/*.cpp') +
    Glob('build/librhi/vk/*.cpp')
)
Export('librhi')

# We build imgui and implot together into a single static library.
# This is also where the imgui config directives live.
env.Append(CPPDEFINES=['IMGUI_DISABLE_OBSOLETE_KEYIO', 'IMGUI_DISABLE_OBSOLETE_FUNCTIONS'])
# Map imgui and implot source trees to their build locations
env.VariantDir('build/libimgui/imgui', 'imgui', duplicate=False)
env.VariantDir('build/libimgui/implot', 'implot', duplicate=False)
libimgui = env.StaticLibrary(
  target = '#build/lib/imgui',
  source = Glob('build/libimgui/imgui/*.cpp') + Glob('build/libimgui/implot/*.cpp')
)
Export('libimgui')


##### Build binaries #####

build_dgpu = True
if (is_tegra and (not os.path.isdir('/usr/local/nvidia-dgpu-support'))):
  build_dgpu = False
  print('DGPU support libraries for Tegra are not installed at /usr/local/nvidia-dgpu-support. DGPU backend will not build.')

SConscript('SConscript-hmdcam', variant_dir = 'build/hmdcam', duplicate = 0)
SConscript('SConscript-canbus-test', variant_dir = 'build/canbus-test', duplicate = 0)
SConscript('SConscript-calibration', variant_dir = 'build/calibration', duplicate = 0)

if not is_tegra:
  # Only build test apps on desktop
  SConscript('SConscript-stereo-geometry', variant_dir = 'build/stereo-geometry', duplicate = 0)

if have_opencv_cuda:
  SConscript('SConscript-debug-client', variant_dir = 'build/debug-client', duplicate = 0)
if (build_dgpu):
  SConscript('SConscript-dgpu-worker', variant_dir = 'build/dgpu-worker', duplicate = 0)
if is_tegra:
  SConscript('SConscript-disparity-test', variant_dir = 'build/disparity-test', duplicate = 0)

# SHM worker benchmark framework
SConscript('SConscript-worker-benchmark', variant_dir = 'build/worker-benchmark', duplicate = 0)

# DepthAI worker (optional)
if os.path.isdir('build/depthai-core/install'):
  SConscript('SConscript-depthai-worker', variant_dir = 'build/depthai-worker', duplicate = 0)
else:
  print('Did not find build artifacts from depthai-core -- the DepthAI worker will be disabled.')
  print('If you want to build the DepthAI worker, run ./build-depthai-core.sh first')

# Eyetracking test harness
if (is_tegra):
  SConscript('SConscript-dla-standalone-test', variant_dir = 'build/dla-standalone-test', duplicate = 0)
  # SConscript('SConscript-eyetracking-test', variant_dir = 'build/eyetracking-test', duplicate = 0)
  SConscript('SConscript-eyetracking-hmd-test', variant_dir = 'build/eyetracking-hmd-test', duplicate = 0)
  SConscript('SConscript-facetracking-hmd-test', variant_dir = 'build/facetracking-hmd-test', duplicate = 0)
  SConscript('SConscript-v4l2-capture-test', variant_dir = 'build/v4l2-capture-test', duplicate = 0)

