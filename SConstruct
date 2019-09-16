import platform
import os

# Environment setup
env = Environment(
  CPPPATH=['/opt/vc/include', '/opt/vc/include/interface/vcos/pthreads', '/opt/vc/include/interface/vmcs_host/linux'],
  CPPFLAGS=['-ggdb', '-std=c++11', '-Wall'],
  LINKFLAGS=['-ggdb'],
  LIBPATH=['/opt/vc/lib'],
)

# Fix for clang colored diagnostics
env['ENV']['TERM'] = os.environ['TERM']

#env.Append(CPPPATH=['.', '#../gameengine', '#../externals/stb'])

env.Decider('MD5-timestamp')
Export('env')

env.Program(
  target = 'hmdcam',
  source = Glob('*.cpp') + Glob('*.c'),
  LIBS=['stdc++', 'mmal', 'mmal_core', 'mmal_util', 'brcmGLESv2', 'brcmEGL', 'bcm_host', 'vcos']
)

