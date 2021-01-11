import platform
import os

# Environment setup
env = Environment(tools = ['clang', 'clangxx', 'link'],
  CPPPATH=['.', 'tegra_mmapi', 'live555/include', 'glm', 'glatter/include', 'glatter/include/glatter', 'imgui', '/usr/include/drm', '/usr/src/tegra_multimedia_api/include', '/usr/src/tegra_multimedia_api/argus/include', '/usr/local/include/opencv4', '/usr/local/cuda/include'],
  CPPFLAGS=['-g', '-Wall'],
  CPPDEFINES=['NO_OPENSSL'],
  CXXFLAGS=['-std=c++11'],
  LINKFLAGS=['-g'],
  LIBPATH=['/usr/lib/aarch64-linux-gnu/tegra', '/usr/local/lib', '/usr/local/cuda/lib64'],
)

# Import locally built Monado static libraries
MONADO_BASE='/mnt/scratch/monado'
MONADO_BUILD=MONADO_BASE + '/build-minimal'

env.Append(
  CPPPATH=[
    MONADO_BUILD + '/src/xrt/include', # Contains generated config headers
    MONADO_BASE + '/src/xrt/include',
    MONADO_BASE + '/src/xrt/auxiliary',
  ]
)
monadoStatics = \
    Glob(MONADO_BUILD + '/src/xrt/targets/common/*.a') + \
    Glob(MONADO_BUILD + '/src/xrt/state_trackers/prober/*.a') + \
    Glob(MONADO_BUILD + '/src/xrt/drivers/*.a') + \
    Glob(MONADO_BUILD + '/src/xrt/state_trackers/oxr/*.a') + \
    [
      MONADO_BUILD + '/src/xrt/auxiliary/libaux_tracking.a',
      MONADO_BUILD + '/src/xrt/auxiliary/libaux_os.a',
      MONADO_BUILD + '/src/xrt/auxiliary/libaux_util.a',
      MONADO_BUILD + '/src/xrt/auxiliary/libaux_math.a' # Needs to be after libaux_tracking.a for symbol resolution.
    ]

# Fix for clang colored diagnostics
env['ENV']['TERM'] = os.environ['TERM']

env.Decider('MD5-timestamp')
Export('env')
liveMediaStatics = [
  'live555/libliveMedia.a',
  'live555/libgroupsock.a',
  'live555/libBasicUsageEnvironment.a',
  'live555/libUsageEnvironment.a']

env.Program(
  target = 'hmdcam',
  source = Glob('*.cpp') + Glob('*.c') + Glob('rhi/*.cpp') + Glob('rhi/gl/*.cpp') + Glob('imgui/*.cpp') + Glob('tegra_mmapi/*.cpp') + liveMediaStatics + monadoStatics,
  LIBS=['z', 'stdc++', 'boost_chrono', 'boost_system', 'GLESv2', 'EGL', 'dl', 'pthread', 'hidapi-libusb', 'usb-1.0', 'udev', 'nvargus', 'nvbuf_utils', 'v4l2', 'opencv_core', 'opencv_imgproc', 'opencv_calib3d', 'opencv_aruco', 'cuda', 'cudart']
)

