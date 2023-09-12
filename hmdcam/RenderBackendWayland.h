#pragma once
#include "RenderBackend.h"
#include <epoxy/egl.h>
#include <wayland-client.h>
#include <wayland-egl.h>
#include "fullscreen-shell-unstable-v1-client-header.h"
#include <map>
#include <string>
#include <vector>
#include "rhi/gl/RHIWindowRenderTargetGL.h"

class RenderBackendWayland;

class WaylandEGLWindowRenderTarget : public RHIWindowRenderTargetGL {
public:
  typedef boost::intrusive_ptr<WaylandEGLWindowRenderTarget> ptr;
  WaylandEGLWindowRenderTarget(RenderBackendWayland*);
  virtual ~WaylandEGLWindowRenderTarget();
  virtual void platformSwapBuffers();

protected:
  RenderBackendWayland* m_backend;
};

class RenderBackendWayland : public RenderBackend {
public:
  RenderBackendWayland();
  virtual ~RenderBackendWayland();

  virtual void init();

  virtual uint32_t surfaceWidth() const { return m_surfaceWidth; }
  virtual uint32_t surfaceHeight() const { return m_surfaceHeight; }
  virtual double refreshRateHz() const { return m_refreshRateHz; }

  virtual EGLDisplay eglDisplay() const { return m_eglDisplay; }
  virtual EGLContext eglContext() const { return m_eglContext; }
  virtual EGLSurface eglSurface() const { return m_eglSurface; }
  virtual EGLConfig eglConfig() const { return m_eglConfig; }

  virtual RHIRenderTarget::ptr windowRenderTarget() const { return m_windowRenderTarget; }

private:
  friend class WaylandEGLWindowRenderTarget;

  uint32_t m_surfaceWidth = 0;
  uint32_t m_surfaceHeight = 0;
  double m_refreshRateHz = 0;

  EGLDisplay m_eglDisplay = EGL_NO_DISPLAY;
  EGLContext m_eglContext = NULL;
  EGLConfig m_eglConfig = NULL;
  EGLSurface m_eglSurface = NULL;

  WaylandEGLWindowRenderTarget::ptr m_windowRenderTarget;

  wl_display* m_wlDisplay = NULL;
  wl_registry* m_wlRegistry = NULL;
  wl_compositor* m_wlCompositor = NULL;

  struct OutputData {
    OutputData(RenderBackendWayland* _owner, uint32_t _wl_name, wl_output* _wl_output) : m_owner(_owner), m_wlName(_wl_name), m_wlOutput(_wl_output) {}

    RenderBackendWayland* m_owner = NULL;
    uint32_t m_wlName = 0;
    wl_output* m_wlOutput = NULL;

    bool update_done = false;

    // wl_output_listener::geometry
    int32_t x = 0;
    int32_t y = 0;
    int32_t physical_width = 0;
    int32_t physical_height = 0;
    int32_t subpixel = 0;
    std::string make;
    std::string model;
    int32_t transform = 0;

    // wl_output_listener::mode
    struct Mode {
      Mode() {}
      Mode(uint32_t _f, int32_t _w, int32_t _h, int32_t _r) : flags(_f), width(_w), height(_h), refresh(_r) {}

      uint32_t flags = 0;
      int32_t width = 0, height = 0;
      int32_t refresh = 0; // milliHertz -- 60hz would be 60000
    };
    std::vector<Mode> modes;

    Mode* currentMode() {
      for (size_t i = 0; i < modes.size(); ++i) {
        if (modes[i].flags & WL_OUTPUT_MODE_CURRENT)
          return &modes[i];
      }
      return NULL;
    }

    // wl_output_listener::scale
    int scaleFactor = 1;
  };

  std::map<uint32_t, OutputData*> m_wlOutputs;

  wl_shell* m_wlShell = NULL;
  wl_shell_surface* m_wlShellSurface = NULL;

  zwp_fullscreen_shell_v1* m_wlFullscreenShell = NULL;

  wl_surface* m_wlSurface;
  wl_egl_window* m_wlEglWindow = NULL;

  void registryHandleGlobal(struct wl_registry *registry, uint32_t name, const char *interface, uint32_t version);
  void registryHandleGlobalRemove( struct wl_registry *registry, uint32_t name);

  void didAddOutput(OutputData*);
  void didRemoveOutput(OutputData*);

  // Shell
  void shellSurfaceHandleConfigure(struct wl_shell_surface *shell_surface, uint32_t edges, int32_t width, int32_t height);

  // Seat
  wl_seat* m_wlSeat = NULL;
  void seatCapabilities(struct wl_seat *wl_seat, uint32_t caps);
  void seatName(struct wl_seat *wl_seat, const char *name);

  // Keyboard
  wl_keyboard* m_wlKeyboard = nullptr;
  struct xkb_context *m_xkbContext = nullptr;
  struct xkb_keymap *m_xkbKeymap = nullptr;
  struct xkb_state *m_xkbState = nullptr;

  void keyboardKeymap(struct wl_keyboard *wl_kbd, uint32_t format, int fd, uint32_t size);
  void keyboardEnter(struct wl_keyboard *wl_kbd, uint32_t serial, struct wl_surface *surf, struct wl_array *keys);
  void keyboardLeave(struct wl_keyboard *wl_kbd, uint32_t serial, struct wl_surface *surf);
  void keyboardKey(struct wl_keyboard *wl_kbd, uint32_t serial, uint32_t time, uint32_t key, uint32_t state);
  void keyboardModifiers(struct wl_keyboard *wl_kbd, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group);
  void keyboardRepeatInfo(struct wl_keyboard *wl_kbd, int32_t rate, int32_t delay);


  // Thunks -- registry
  static struct wl_registry_listener registry_listener;
  static void registry_handle_global(void *data, struct wl_registry *registry, uint32_t name, const char *interface, uint32_t version) {
    reinterpret_cast<RenderBackendWayland*>(data)->registryHandleGlobal(registry, name, interface, version);
  }
  static void registry_handle_global_remove(void *data, struct wl_registry *registry, uint32_t name) {
    reinterpret_cast<RenderBackendWayland*>(data)->registryHandleGlobalRemove(registry, name);
  }

  // Thunks -- shell
  static struct wl_shell_surface_listener shell_surface_listener;
  static void shell_surface_handle_configure(void *data, struct wl_shell_surface *shell_surface, uint32_t edges, int32_t width, int32_t height) {
    reinterpret_cast<RenderBackendWayland*>(data)->shellSurfaceHandleConfigure(shell_surface, edges, width, height);
  }

  // Thunks -- seat
  static struct wl_seat_listener seat_listener;
  static void seat_capabilities(void *data, struct wl_seat *wl_seat, uint32_t caps) {
    reinterpret_cast<RenderBackendWayland*>(data)->seatCapabilities(wl_seat, caps);
  }

  static void seat_name(void *data, struct wl_seat *wl_seat, const char *name) {
    reinterpret_cast<RenderBackendWayland*>(data)->seatName(wl_seat, name);
  }

  // Thunks -- keyboard
  static struct wl_keyboard_listener keyboard_listener;
  static void keyboard_keymap(void *data, struct wl_keyboard *wl_kbd, uint32_t format, int fd, uint32_t size) {
    reinterpret_cast<RenderBackendWayland*>(data)->keyboardKeymap(wl_kbd, format, fd, size);
  }
  static void keyboard_enter(void *data, struct wl_keyboard *wl_kbd, uint32_t serial, struct wl_surface *surf, struct wl_array *keys) {
    reinterpret_cast<RenderBackendWayland*>(data)->keyboardEnter(wl_kbd, serial, surf, keys);
  }
  static void keyboard_leave(void *data, struct wl_keyboard *wl_kbd, uint32_t serial, struct wl_surface *surf) {
    reinterpret_cast<RenderBackendWayland*>(data)->keyboardLeave(wl_kbd, serial, surf);
  }
  static void keyboard_key(void *data, struct wl_keyboard *wl_kbd, uint32_t serial, uint32_t time, uint32_t key, uint32_t state) {
    reinterpret_cast<RenderBackendWayland*>(data)->keyboardKey(wl_kbd, serial, time, key, state);
  }
  static void keyboard_modifiers(void *data, struct wl_keyboard *wl_kbd, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group) {
    reinterpret_cast<RenderBackendWayland*>(data)->keyboardModifiers(wl_kbd, serial, mods_depressed, mods_latched, mods_locked, group);
  }
  static void keyboard_repeat_info(void *data, struct wl_keyboard *wl_kbd, int32_t rate, int32_t delay) {
    reinterpret_cast<RenderBackendWayland*>(data)->keyboardRepeatInfo(wl_kbd, rate, delay);
  }

  // Thunks -- output
  static void outputGeometry(void* _data, struct wl_output *wl_output, int32_t x, int32_t y, int32_t physical_width, int32_t physical_height, int32_t subpixel, const char *make, const char *model, int32_t transform);
  static void outputMode(void* _data, struct wl_output *wl_output, uint32_t flags, int32_t width, int32_t height, int32_t refresh);
  static void outputDone(void* _data, struct wl_output *wl_output);
  static void outputScale(void* _data, struct wl_output *wl_output, int32_t factor);
  static struct wl_output_listener output_listener;
};

