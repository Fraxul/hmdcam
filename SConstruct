import platform
import os

# Environment setup
env = Environment(
  CPPPATH=['/usr/include/drm'],
  CPPFLAGS=['-ggdb', '-std=c++11', '-Wall'],
  LINKFLAGS=['-ggdb'],
  #LIBPATH=[],
)

# Fix for clang colored diagnostics
env['ENV']['TERM'] = os.environ['TERM']

#env.Append(CPPPATH=['.', '#../gameengine', '#../externals/stb'])

env.Decider('MD5-timestamp')
Export('env')

env.Program(
  target = 'hmdcam',
  source = Glob('*.cpp') + Glob('*.c') + ['openhmd/libopenhmd.a'],
  LIBS=['stdc++', 'GLESv2', 'EGL', 'dl', 'pthread', 'hidapi-hidraw']
)

