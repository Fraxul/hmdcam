#include "RenderBackendWayland.h"
#include <epoxy/egl.h>
#include <wayland-client.h>
#include <wayland-egl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>

#define CHECK_PTR(x) if ((x) == nullptr) { fprintf(stderr, "%s:%d: %s failed (%s)\n", __FILE__, __LINE__, #x, strerror(errno)); abort(); }

#define DRM_CHECK(x) if ((x) != 0) { fprintf(stderr, "%s:%d: %s failed (%s)\n", __FILE__, __LINE__, #x, strerror(errno)); abort(); }
#define EGL_CHECK_BOOL(x) if (!(x)) { fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, eglGetError()); abort(); }

RenderBackend* createWaylandBackend() { return new RenderBackendWayland(); }

struct wl_registry_listener RenderBackendWayland::registry_listener = {
    registry_handle_global,
    registry_handle_global_remove
};

static void shell_surface_handle_ping(void *data, struct wl_shell_surface *wlShellSurface, uint32_t serial) {
  wl_shell_surface_pong(wlShellSurface, serial);
}

void RenderBackendWayland::shellSurfaceHandleConfigure(struct wl_shell_surface *shell_surface, uint32_t edges, int32_t width, int32_t height) {
  if (m_wlEglWindow)
    wl_egl_window_resize(m_wlEglWindow, width, height, 0, 0);

  printf("shellSurfaceHandleConfigure: %ux%u => %dx%d edges=%u\n", m_surfaceWidth, m_surfaceHeight, width, height, edges);

  m_surfaceWidth = width;
  m_surfaceHeight = height;
}

struct wl_shell_surface_listener RenderBackendWayland::shell_surface_listener = {
    shell_surface_handle_ping,
    shell_surface_handle_configure,
};

RenderBackendWayland::RenderBackendWayland() { }

void RenderBackendWayland::init() {
  CHECK_PTR(m_wlDisplay = wl_display_connect(NULL));
  CHECK_PTR(m_wlRegistry = wl_display_get_registry(m_wlDisplay));

  wl_registry_add_listener(m_wlRegistry, &registry_listener, this);
  wl_display_dispatch(m_wlDisplay);
  wl_display_roundtrip(m_wlDisplay);

  CHECK_PTR(m_wlCompositor);
  CHECK_PTR(m_wlSurface = wl_compositor_create_surface(m_wlCompositor));

  if (m_wlFullscreenShell) {
    zwp_fullscreen_shell_v1_present_surface(m_wlFullscreenShell, m_wlSurface, ZWP_FULLSCREEN_SHELL_V1_PRESENT_METHOD_DEFAULT, /*output=*/ NULL);
  } else if (m_wlShell) {
    CHECK_PTR(m_wlShellSurface = wl_shell_get_shell_surface(m_wlShell, m_wlSurface));
    wl_shell_surface_add_listener(m_wlShellSurface, &shell_surface_listener, this);
    wl_shell_surface_set_toplevel(m_wlShellSurface);
    wl_shell_surface_set_fullscreen(m_wlShellSurface, WL_SHELL_SURFACE_FULLSCREEN_METHOD_DEFAULT, 0, NULL);
  } else {
    assert(false && "Couldn't get a shell interface (wl_shell or zwp_fullscreen_shell_v1)");
  }

  OutputData::Mode* currentMode = NULL;
  while (true) {
    // TODO output selection in multi-output case?
    if (!m_wlOutputs.empty()) {
      currentMode = m_wlOutputs.begin()->second->currentMode();
    }
    if (currentMode)
      break;
    // no mode/display yet, dispatch events
    printf("Waiting for display/mode enumeration\n");
    wl_display_dispatch(m_wlDisplay);
  }

  m_surfaceWidth = currentMode->width;
  m_surfaceHeight = currentMode->height;
  m_refreshRateHz = static_cast<double>(currentMode->refresh) / 1000.0;
  CHECK_PTR(m_wlEglWindow = wl_egl_window_create(m_wlSurface, m_surfaceWidth, m_surfaceHeight));

  wl_display_roundtrip(m_wlDisplay);

  CHECK_PTR(m_eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_WAYLAND_EXT, (EGLNativeDisplayType) m_wlDisplay, NULL));
  EGL_CHECK_BOOL(eglInitialize(m_eglDisplay, NULL, NULL));

  EGLint cfg_attr[] = {
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
    EGL_RED_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8,
    EGL_ALPHA_SIZE, 8,
    EGL_NONE
  };

  int n;
  EGL_CHECK_BOOL(eglChooseConfig(m_eglDisplay, cfg_attr, &m_eglConfig, 1, &n));

  CHECK_PTR(m_eglSurface = eglCreatePlatformWindowSurfaceEXT(m_eglDisplay, m_eglConfig, m_wlEglWindow, /*attrList=*/ nullptr));

  EGLint surfaceWidth = 0, surfaceHeight = 0;
  eglQuerySurface(m_eglDisplay, m_eglSurface, EGL_WIDTH, &surfaceWidth);
  eglQuerySurface(m_eglDisplay, m_eglSurface, EGL_HEIGHT, &surfaceHeight);
  printf("eglCreatePlatformWindowSurfaceEXT: %d x %d\n", surfaceWidth, surfaceHeight);

  // Set optional surface attributes
  eglSurfaceAttrib(m_eglDisplay, m_eglSurface, EGL_SWAP_BEHAVIOR, EGL_BUFFER_DESTROYED);

  EGLint ctx_attr[] = {
    EGL_CONTEXT_CLIENT_VERSION, 3,
    EGL_NONE
  };

  eglBindAPI(EGL_OPENGL_API);
  CHECK_PTR(m_eglContext = eglCreateContext(m_eglDisplay, m_eglConfig, EGL_NO_CONTEXT, ctx_attr));
  EGL_CHECK_BOOL(eglMakeCurrent(m_eglDisplay, m_eglSurface, m_eglSurface, m_eglContext));

  wl_display_roundtrip(m_wlDisplay);

  m_windowRenderTarget = new WaylandEGLWindowRenderTarget(this);
  m_windowRenderTarget->platformSetUpdatedWindowDimensions(m_surfaceWidth, m_surfaceHeight);
}


// Registry handling static function
void RenderBackendWayland::registryHandleGlobal(struct wl_registry *registry, uint32_t name, const char *interface, uint32_t version) {
  printf("registryHandleGlobal: [%u] %s version=%u\n", name, interface, version);
  if (strcmp(interface, "wl_compositor") == 0) {
    m_wlCompositor = (wl_compositor*) wl_registry_bind(registry, name, &wl_compositor_interface, 1);
  } else if (!strcmp(interface, "zwp_fullscreen_shell_v1")) {
    m_wlFullscreenShell = (zwp_fullscreen_shell_v1*) wl_registry_bind(registry, name, &zwp_fullscreen_shell_v1_interface, 1);
  } else if (strcmp(interface, "wl_shell") == 0) {
    m_wlShell = (wl_shell*) wl_registry_bind(registry, name, &wl_shell_interface, 1);
  } else if (!strcmp(interface, "wl_output")) {
    wl_output* output = (wl_output*) wl_registry_bind(registry, name, &wl_output_interface, 1);
    auto outputData = new OutputData(this, name, output);
    m_wlOutputs[name] = outputData;
    didAddOutput(outputData);
#if 0
  } else if (strcmp(interface, "zwp_linux_dmabuf_v1") == 0) {
    d->wlDmabuf = wl_registry_bind(registry, name, &zwp_linux_dmabuf_v1_interface, 3);
    memset(d->formatModifiers_XRGB8, 0xff, sizeof(d->formatModifiers_XRGB8));
    memset(d->formatModifiers_ARGB8, 0xff, sizeof(d->formatModifiers_ARGB8));
    zwp_linux_dmabuf_v1_add_listener(d->wlDmabuf, &dmabuf_listener, d);
  } else if (strcmp(interface, "zwp_linux_explicit_synchronization_v1") == 0) {
    d->wlExplicitSync = wl_registry_bind(registry, name, &zwp_linux_explicit_synchronization_v1_interface, 1);
#endif
  }
}

void RenderBackendWayland::registryHandleGlobalRemove( struct wl_registry *registry, uint32_t name) {
  printf("registryHandleGlobalRemove: remove %u\n", name);
  auto it = m_wlOutputs.find(name);
  if (it != m_wlOutputs.end()) {
    didRemoveOutput(it->second);
    delete it->second;
    m_wlOutputs.erase(it);
  }
}

/*static*/ void RenderBackendWayland::outputGeometry(void* _data, struct wl_output *wl_output, int32_t x, int32_t y, int32_t physical_width, int32_t physical_height, int32_t subpixel, const char *make, const char *model, int32_t transform) {
  OutputData* data = reinterpret_cast<OutputData*>(_data);
  assert(data->m_wlOutput == wl_output);

  data->x = x;
  data->y = y;
  data->physical_width = physical_width;
  data->physical_height = physical_height;
  data->subpixel = subpixel;
  data->make = make;
  data->model = model;
  data->transform = transform;

  printf("wl_output[%u] geometry: x=%d y=%d physW=%d physH=%d subpixel=%u make=%s model=%s transform=%d\n",
    data->m_wlName, x, y, physical_width, physical_height, subpixel, make, model, transform);
}

/*static*/ void RenderBackendWayland::outputMode(void* _data, struct wl_output *wl_output, uint32_t flags, int32_t width, int32_t height, int32_t refresh) {
  OutputData* data = reinterpret_cast<OutputData*>(_data);
  assert(data->m_wlOutput == wl_output);

  printf("wl_output[%u] mode: flags=0x%x w=%d height=%d refresh=%.3gHz %s%s\n",
    data->m_wlName, flags, width, height, static_cast<float>(refresh) / 1000.0f,
    (flags & WL_OUTPUT_MODE_CURRENT) ? " current" : "", (flags & WL_OUTPUT_MODE_PREFERRED) ? " preferred" : "");

  data->modes.push_back(OutputData::Mode(flags, width, height, refresh));
}

/*static*/ void RenderBackendWayland::outputDone(void* _data, struct wl_output *wl_output) {
  OutputData* data = reinterpret_cast<OutputData*>(_data);
  assert(data->m_wlOutput == wl_output);

  printf("wl_output[%u] mode info done\n", data->m_wlName);
  data->update_done = true;
}

/*static*/ void RenderBackendWayland::outputScale(void* _data, struct wl_output *wl_output, int32_t factor) {
  OutputData* data = reinterpret_cast<OutputData*>(_data);
  assert(data->m_wlOutput == wl_output);

  printf("wl_output[%u] output scale factor: %d\n", data->m_wlName, factor);
  data->scaleFactor = factor;
}


struct wl_output_listener RenderBackendWayland::output_listener = {
    outputGeometry,
    outputMode,
    outputDone,
    outputScale
};

void RenderBackendWayland::didAddOutput(OutputData* outputData) {
  printf("didAddOutput(%u) = %p\n", outputData->m_wlName, outputData->m_wlOutput);
  wl_output_add_listener(outputData->m_wlOutput, &output_listener, outputData);
}

void RenderBackendWayland::didRemoveOutput(OutputData* outputData) {
  printf("didRemoveOutput(%u)\n", outputData->m_wlName);
}

RenderBackendWayland::~RenderBackendWayland() {

}

WaylandEGLWindowRenderTarget::WaylandEGLWindowRenderTarget(RenderBackendWayland* rb) : m_backend(rb) {

}

WaylandEGLWindowRenderTarget::~WaylandEGLWindowRenderTarget() {

}

void WaylandEGLWindowRenderTarget::platformSwapBuffers() {
  eglSwapBuffers(m_backend->m_eglDisplay, m_backend->m_eglSurface);
  wl_display_dispatch_pending(m_backend->m_wlDisplay);
}

