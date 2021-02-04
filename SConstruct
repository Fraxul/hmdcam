import platform
import os

is_tegra = os.access("/dev/tegra_camera_ctrl", os.F_OK)

if is_tegra:
  # Environment setup
  env = Environment(tools = ['clang', 'clangxx', 'link', 'cuda'], toolpath=['scons-tools'],
    CPPPATH=['#tegra_mmapi', '#live555/include', '/usr/include/drm', '/usr/src/tegra_multimedia_api/include', '/usr/src/tegra_multimedia_api/argus/include', '/usr/local/cuda/include'],
    LIBPATH=['/usr/lib/aarch64-linux-gnu/tegra', '/usr/local/lib', '/usr/local/cuda/lib64'],
    CUDA_SDK_PATH='/usr/local/cuda'
  )

else:
  # Reduced environment for non-tegra
  # TODO CUDA path
  env = Environment(tools = ['clang', 'clangxx', 'link', 'cuda'], toolpath=['scons-tools'],
    CPPPATH=[],
    LIBPATH=[],
    CUDA_SDK_PATH='/usr/local/cuda'
  )

# Common env
env.Append(
  CPPPATH=['#.', '#glm', '#glatter/include', '#glatter/include/glatter', '#imgui',  '/usr/local/include/opencv4'],
  CPPFLAGS=['-g', '-Wall'],
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

