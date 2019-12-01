import platform
import os

# Environment setup
env = Environment(
  CPPPATH=['.', 'glm', 'glatter/include', 'glatter/include/glatter', '/usr/include/drm', '/usr/src/nvidia/tegra_multimedia_api/include', '/usr/src/nvidia/tegra_multimedia_api/argus/include'],
  CPPFLAGS=['-ggdb', '-std=c++11', '-Wall'],
  LINKFLAGS=['-ggdb'],
  LIBPATH=['/usr/lib/aarch64-linux-gnu/tegra'],
)

# Fix for clang colored diagnostics
env['ENV']['TERM'] = os.environ['TERM']

env.Decider('MD5-timestamp')
Export('env')

env.Program(
  target = 'hmdcam',
  source = Glob('*.cpp') + Glob('*.c') + Glob('rhi/*.cpp') + Glob('rhi/gl/*.cpp') + ['openhmd/libopenhmd.a'],
  LIBS=['stdc++', 'boost_chrono', 'boost_system', 'GLESv2', 'EGL', 'dl', 'pthread', 'hidapi-hidraw', 'nvargus', 'nvbuf_utils', 'opencv_core', 'opencv_imgproc', 'opencv_calib3d']
)

