#include "GLUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

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

void createFBO(int width, int height, GLuint* outFBO, GLuint* outColorTex, GLuint* outDepthTex) {
  GL(glGenTextures(1, outColorTex));
  if (outDepthTex) {
    GL(glGenTextures(1, outDepthTex));
  }
  GL(glGenFramebuffers(1, outFBO));
  
  GL(glBindTexture(GL_TEXTURE_2D, *outColorTex));
  //GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
  GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, NULL));
  GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
  GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
 
  if (outDepthTex) { 
    GL(glBindTexture(GL_TEXTURE_2D, *outDepthTex));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glBindTexture(GL_TEXTURE_2D, 0));
  }
    
  GL(glBindFramebuffer(GL_FRAMEBUFFER, *outFBO));
  GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *outColorTex, 0));

  if (outDepthTex) {
    GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, *outDepthTex, 0));
  }

  GLenum status = GL(glCheckFramebufferStatus(GL_FRAMEBUFFER));

  if(status != GL_FRAMEBUFFER_COMPLETE){
    die("Failed to create %u x %u fbo: %x\n", width, height, status);
  }
  GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

