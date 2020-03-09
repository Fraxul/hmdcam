import platform
import os

# Environment setup
env = Environment(tools = ['clang', 'clangxx', 'link'],
  CPPPATH=['.', 'glm', 'glatter/include', 'glatter/include/glatter', 'imgui', '/usr/include/drm', '/usr/src/tegra_multimedia_api/include', '/usr/src/tegra_multimedia_api/argus/include', '/usr/local/include/opencv4'],
  CPPFLAGS=['-g', '-Wall'],
  CXXFLAGS=['-std=c++11'],
  LINKFLAGS=['-g'],
  LIBPATH=['/usr/lib/aarch64-linux-gnu/tegra', '/usr/local/lib'],
)

# Fix for clang colored diagnostics
env['ENV']['TERM'] = os.environ['TERM']

env.Decider('MD5-timestamp')
Export('env')

env.Program(
  target = 'hmdcam',
  source = Glob('*.cpp') + Glob('*.c') + Glob('rhi/*.cpp') + Glob('rhi/gl/*.cpp') + Glob('imgui/*.cpp') + ['openhmd/libopenhmd.a'],
  LIBS=['z', 'stdc++', 'boost_chrono', 'boost_system', 'GLESv2', 'EGL', 'dl', 'pthread', 'hidapi-hidraw', 'udev', 'nvargus', 'nvbuf_utils', 'opencv_core', 'opencv_imgproc', 'opencv_calib3d', 'opencv_aruco']
)

