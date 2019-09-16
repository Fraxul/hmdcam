#include "GLUtils.h"
#include <stdio.h>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglext_brcm.h>

void checkGLError(const char* op, const char* file, int line) {
  GLint error = glGetError();
  if (error)
    fprintf(stderr, "after %s (%s:%d) glError (0x%x)\n", op, file, line, error);
}


void checkEGLError(const char* op, const char* file, int line) {
  GLint error = eglGetError();
  if (error)
    fprintf(stderr, "after %s (%s:%d) eglError (0x%x)\n", op, file, line, error);
}

GLuint compileShader(const char* vertexSource, const char* fragmentSource) {
	GLuint vertexShader;
	GLuint fragmentShader;
	GLint compiled;
		 
	vertexShader = GL(glCreateShader(GL_VERTEX_SHADER));
	GL(glShaderSource(vertexShader, 1, (const GLchar **) &vertexSource, NULL));

	GL(glCompileShader(vertexShader));
	GL(glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &compiled));

	if (!compiled) {
    printf("Vertex shader didn't compile\n");
    abort();
  }

	fragmentShader = GL(glCreateShader(GL_FRAGMENT_SHADER));
	GL(glShaderSource(fragmentShader, 1, (const GLchar **) &fragmentSource, NULL));

	GL(glCompileShader(fragmentShader));
	GL(glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &compiled));

	if (!compiled) {
    printf("Fragment shader didn't compile\n");
  }

	GLuint programObject = GL(glCreateProgram());
	GL(glAttachShader(programObject, vertexShader));
	GL(glAttachShader(programObject, fragmentShader));

	GL(glLinkProgram(programObject));

	GLint linked;
	GL(glGetProgramiv(programObject, GL_LINK_STATUS, &linked));

  if(!linked) 
  {
    printf("can't link shader !!\n");
    GLint infoLen = 0;
    GL(glGetProgramiv(programObject, GL_INFO_LOG_LENGTH, &infoLen));

    if(infoLen > 1)
    {
      char* infoLog = (char *) malloc(sizeof(char) * infoLen);
      glGetProgramInfoLog(programObject, infoLen, NULL, infoLog);
      if ( infoLen ){
        printf("error log :\n");
        printf("%s\n",infoLog);
      }

      free(infoLog);
    }
    glDeleteProgram(programObject);
    abort();
  }

  return programObject;
}

