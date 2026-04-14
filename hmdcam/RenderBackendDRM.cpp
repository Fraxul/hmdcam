#include "RenderBackendDRM.h"
#include <epoxy/egl.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>

#define DRM_CHECK(x) if ((x) != 0) { fprintf(stderr, "%s:%d: %s failed (%s)\n", __FILE__, __LINE__, #x, strerror(errno)); abort(); }
#define DRM_CHECK_PTR(x) if ((x) == nullptr) { fprintf(stderr, "%s:%d: %s failed (%s)\n", __FILE__, __LINE__, #x, strerror(errno)); abort(); }
#define EGL_CHECK_BOOL(x) if (!(x)) { fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, eglGetError()); abort(); }
#define CheckExtension(extStr) if (!epoxy_has_egl_extension(m_eglDisplay, extStr)) { fprintf(stderr, "Missing required EGL extension %s.\n\nExtension string: %s\n\n", extStr, eglQueryString(m_eglDisplay, EGL_EXTENSIONS)); abort(); }


// Allow building on old libepoxy (1.4.3-1) on Jetpack < 5.0
#ifndef EGL_DRM_MASTER_FD_EXT
#define EGL_DRM_MASTER_FD_EXT 0x333C
#endif

RenderBackend* createDRMBackend() { return new RenderBackendDRM(); }

// Look up the ID of a named property on a DRM object. Returns 0 if not found.
static uint32_t drmGetPropertyId(int fd, uint32_t objectId, uint32_t objectType, const char* name) {
  drmModeObjectProperties* props = drmModeObjectGetProperties(fd, objectId, objectType);
  if (!props)
    return 0;
  uint32_t id = 0;
  for (uint32_t i = 0; i < props->count_props; ++i) {
    drmModePropertyRes* prop = drmModeGetProperty(fd, props->props[i]);
    if (!prop)
      continue;

    if (!strcmp(prop->name, name))
      id = prop->prop_id;

    drmModeFreeProperty(prop);

    if (id)
      break;
  }
  drmModeFreeObjectProperties(props);
  return id;
}

static const char* drmConnectorTypeToString(int c) {
  switch (c) {
    default:
    case DRM_MODE_CONNECTOR_Unknown: return "unknown";
    case DRM_MODE_CONNECTOR_VGA: return "VGA";
    case DRM_MODE_CONNECTOR_DVII: return "DVI-I";
    case DRM_MODE_CONNECTOR_DVID: return "DVI-D";
    case DRM_MODE_CONNECTOR_DVIA: return "DVI-A";
    case DRM_MODE_CONNECTOR_Composite: return "Composite";
    case DRM_MODE_CONNECTOR_SVIDEO: return "S-Video";
    case DRM_MODE_CONNECTOR_LVDS: return "LVDS";
    case DRM_MODE_CONNECTOR_Component: return "Component";
    case DRM_MODE_CONNECTOR_9PinDIN: return "9-pin DIN";
    case DRM_MODE_CONNECTOR_DisplayPort: return "DP";
    case DRM_MODE_CONNECTOR_HDMIA: return "HDMI-A";
    case DRM_MODE_CONNECTOR_HDMIB: return "HDMI-B";
    case DRM_MODE_CONNECTOR_TV: return "TV";
    case DRM_MODE_CONNECTOR_eDP: return "eDP";
    case DRM_MODE_CONNECTOR_VIRTUAL: return "Virtual";
    case DRM_MODE_CONNECTOR_DSI: return "DSI";
    case DRM_MODE_CONNECTOR_DPI: return "DPI";
  };
}

RenderBackendDRM::RenderBackendDRM() { }

void RenderBackendDRM::init() {
  // This initialization sequence closely follows NVIDIA CUDA sample code.
  // available at https://github.com/NVIDIA/cuda-samples
  // Revision referenced: b312abaa (Feb 3 2022)
  // File referenced: Samples/5_Domain_Specific/simpleGLES_EGLOutput/graphics_interface_egloutput_via_egl.c

  memset(&m_drmFb, 0, sizeof(m_drmFb));

  EGLint deviceCount = 0;
  EGL_CHECK_BOOL(eglQueryDevicesEXT(0, NULL, &deviceCount));
  if (!deviceCount) {
    fprintf(stderr, "No EGL devices returned\n");
    abort();
  }

  {
    EGLDeviceEXT* eglDevices = new EGLDeviceEXT[deviceCount];
    EGL_CHECK_BOOL(eglQueryDevicesEXT(deviceCount, eglDevices, &deviceCount));

    for (int i = 0; i < deviceCount; ++i) {
      const char* drmName = eglQueryDeviceStringEXT(eglDevices[i], EGL_DRM_DEVICE_FILE_EXT);
      fprintf(stderr, "EGL device [%d]: DRM file %s\n", i, drmName);
      if (!drmName)
        continue;

      if (!strcmp(drmName, "drm-nvdc")) {
        m_drmFd = drmOpen(drmName, NULL);
      } else {
        m_drmFd = open(drmName, O_RDWR, 0);
      }
      if (m_drmFd <= 0) {
        fprintf(stderr, "Unable to open DRM devices %s\n", drmName);
        continue;
      }

      m_eglDevice = eglDevices[i];
      fprintf(stderr, " -- Opened DRM device for EGL device %d\n", i);
      break;
    }

    delete[] eglDevices;

    if (!m_eglDevice) {
      fprintf(stderr, "Unable to open any DRM device.\n");
      abort();
    }
  }

  DRM_CHECK(drmSetClientCap(m_drmFd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1));
  DRM_CHECK(drmSetClientCap(m_drmFd, DRM_CLIENT_CAP_ATOMIC, 1));

  DRM_CHECK_PTR(m_drmResources = drmModeGetResources(m_drmFd));

    // Parse connector info
  for (int connIdx = 0; connIdx < m_drmResources->count_connectors; ++connIdx) {
    drmModeConnector* conn = drmModeGetConnector(m_drmFd, m_drmResources->connectors[connIdx]);
    if (!conn) {
      fprintf(stderr, "Connector %d: drmModeGetConnector returned NULL, skipping\n", connIdx);
      continue;
    }

    fprintf(stderr, "Connector %d: %s-%d\n", connIdx, drmConnectorTypeToString(conn->connector_type), conn->connector_type_id);

    // Skip if not connected
    if (conn->connection != DRM_MODE_CONNECTED) {
      fprintf(stderr, " -- not connected\n");
      drmModeFreeConnector(conn);
      continue;
    }

    m_drmConnector = conn;
    break;
  }

  if (!m_drmConnector) {
    fprintf(stderr, "No valid DRM connectors found\n");
    abort();
  }

  // Find an encoder
  if (m_drmConnector->encoder_id) {
    fprintf(stderr, "Reusing existing encoder ID %d\n", m_drmConnector->encoder_id);
    m_drmEncoder = drmModeGetEncoder(m_drmFd, m_drmConnector->encoder_id);
  } else {
    fprintf(stderr, "Selecting encoder index %d / ID %d\n", 0, m_drmConnector->encoders[0]);
    m_drmEncoder = drmModeGetEncoder(m_drmFd, m_drmConnector->encoders[0]);
  }

  if (!m_drmEncoder) {
    fprintf(stderr, "Can't map encoder\n");
    abort();
  }

  // Find a CRTC we can use for this encoder
  if (m_drmEncoder->crtc_id) {
    fprintf(stderr, "Reusing existing CRTC ID %d\n", m_drmEncoder->crtc_id);
    m_drmCrtc = drmModeGetCrtc(m_drmFd, m_drmEncoder->crtc_id);
    for (int crtcIdx = 0; crtcIdx < m_drmResources->count_crtcs; ++crtcIdx) {
      if (m_drmResources->crtcs[crtcIdx] == m_drmEncoder->crtc_id) {
        m_drmCrtcPipe = crtcIdx;
        break;
      }
    }
  } else {
    for (int crtcIdx = 0; crtcIdx < m_drmResources->count_crtcs; ++crtcIdx) {
      if (m_drmEncoder->possible_crtcs & (1 << crtcIdx)) {
        fprintf(stderr, "Selecting CRTC index %d / ID %d\n", crtcIdx, m_drmResources->crtcs[crtcIdx]);
        m_drmCrtc = drmModeGetCrtc(m_drmFd, m_drmResources->crtcs[crtcIdx]);
        m_drmCrtcPipe = crtcIdx;
        break;
      }
    }
  }

  if (!m_drmCrtc) {
    fprintf(stderr, "Can't map CRTC\n");
    abort();
  }

  {
    int maxRefresh = 0;
    char* maxRefreshStr = getenv("MAX_REFRESH_RATE");
    if (maxRefreshStr)
      maxRefresh = atoi(maxRefreshStr);
    if (maxRefresh) {
      fprintf(stderr, "MAX_REFRESH_RATE env set, capping to %d hz\n", maxRefresh);
    }


    std::vector<int> modeSort;
    for (int modeIdx = 0; modeIdx < m_drmConnector->count_modes; ++modeIdx) {
      if (maxRefresh && m_drmConnector->modes[modeIdx].vrefresh > maxRefresh) {
        const auto& mode = m_drmConnector->modes[modeIdx];
        printf(" -- Skipping mode [%d] %dx%d@%d%s due to refresh rate cap\n", modeIdx, mode.hdisplay, mode.vdisplay, mode.vrefresh, (mode.type & DRM_MODE_TYPE_PREFERRED) ? " (preferred)" : "");
        continue;
      }
      modeSort.push_back(modeIdx);
    }
    std::sort(modeSort.begin(), modeSort.end(), [conn = m_drmConnector](int lmi, int rmi) {
      const auto& lm = conn->modes[lmi];
      const auto& rm = conn->modes[rmi];
      // preferred mode first
      if ((lm.type & DRM_MODE_TYPE_PREFERRED) && (!(rm.type & DRM_MODE_TYPE_PREFERRED))) return true;
      if (!(lm.type & DRM_MODE_TYPE_PREFERRED) && (rm.type & DRM_MODE_TYPE_PREFERRED)) return false;
      // then higher resolution
      if (lm.hdisplay > rm.hdisplay) return true;
      if (lm.hdisplay < rm.hdisplay) return false;
      if (lm.vdisplay > rm.vdisplay) return true;
      if (lm.vdisplay < rm.vdisplay) return false;
      // then higher refresh
      if (lm.vrefresh > rm.vrefresh) return true;
      if (lm.vrefresh < rm.vrefresh) return false;
      return false;
    });

    printf("Modes (sorted):\n");
    for (int modeIdx : modeSort) {
      const auto& mode = m_drmConnector->modes[modeIdx];
      printf("[%d] %dx%d@%d %s\n", modeIdx, mode.hdisplay, mode.vdisplay, mode.vrefresh, (mode.type & DRM_MODE_TYPE_PREFERRED) ? "(preferred)" : "");
    }

    // Pick the first usable mode out of the sort-list
    m_drmModeIdx = modeSort[0];
    m_drmModeInfo = &(m_drmConnector->modes[m_drmModeIdx]);
    m_surfaceWidth = m_drmModeInfo->hdisplay;
    m_surfaceHeight = m_drmModeInfo->vdisplay;

    m_refreshRateHz = m_drmModeInfo->vrefresh;

    printf("Selected mode %d:\n", m_drmModeIdx);
    printf("  Name: %s\n", m_drmModeInfo->name);
    printf("  Pixel Clock: %u kHz\n", m_drmModeInfo->clock);
    printf("  Refresh (approx): %u hz\n", m_drmModeInfo->vrefresh);
    printf("  H Display: %u  SyncStart: %u  End: %u  Total: %u  Skew: %u\n", m_drmModeInfo->hdisplay, m_drmModeInfo->hsync_start, m_drmModeInfo->hsync_end, m_drmModeInfo->htotal, m_drmModeInfo->hskew);
    printf("  V Display: %u  SyncStart: %u  End: %u  Total: %u  Scan: %u\n", m_drmModeInfo->vdisplay, m_drmModeInfo->vsync_start, m_drmModeInfo->vsync_end, m_drmModeInfo->vtotal, m_drmModeInfo->vscan);
    printf("  Flags: 0x%x\n", m_drmModeInfo->flags);
    printf("  Type: %u\n", m_drmModeInfo->type);

    // Compute precise refresh rate from timing parameters
    m_refreshRateHz = (m_drmModeInfo->clock /*kHz*/ * 1000.0)  / static_cast<double>(m_drmModeInfo->htotal * m_drmModeInfo->vtotal);
    if (fabs(1.0 - (m_refreshRateHz / static_cast<double>(m_drmModeInfo->vrefresh))) > 0.05) { // 5% tolerance
      printf("WARNING: Computed precise vrefresh timing (%g hz) doesn't line up with provided approximate timing (%u hz), using the approximate.\n", m_refreshRateHz, m_drmModeInfo->vrefresh);
      m_refreshRateHz = m_drmModeInfo->vrefresh;
    } else {
      printf("  Computed refresh (precise): %g hz\n", m_refreshRateHz);
    }
  }
  if (m_drmModeIdx < 0){
    fprintf(stderr, "Couldn't find a usable mode.\n");
    abort();
  }

  // Find the primary plane associated with this CRTC
  {
    drmModePlaneRes* planeRes = nullptr;
    DRM_CHECK_PTR(planeRes = drmModeGetPlaneResources(m_drmFd));

    for (uint32_t i = 0; i < planeRes->count_planes; ++i) {
      drmModePlane* plane = drmModeGetPlane(m_drmFd, planeRes->planes[i]);
      if (!plane)
        continue;

      // Check if this plane can be used with our CRTC
      if (!(plane->possible_crtcs & (1 << m_drmCrtcPipe))) {
        drmModeFreePlane(plane);
        continue;
      }

      // Check if this is the primary plane by reading its "type" property
      drmModeObjectProperties* planeProps = drmModeObjectGetProperties(m_drmFd, plane->plane_id, DRM_MODE_OBJECT_PLANE);
      if (!planeProps) {
        drmModeFreePlane(plane);
        continue;
      }

      bool isPrimary = false;
      for (uint32_t j = 0; j < planeProps->count_props; ++j) {
        drmModePropertyRes* prop = drmModeGetProperty(m_drmFd, planeProps->props[j]);
        if (prop) {
          if (!strcmp(prop->name, "type") && planeProps->prop_values[j] == DRM_PLANE_TYPE_PRIMARY)
            isPrimary = true;
          drmModeFreeProperty(prop);
        }
        if (isPrimary)
          break;
      }
      drmModeFreeObjectProperties(planeProps);

      if (isPrimary) {
        m_drmPlane = plane;
        fprintf(stderr, "Selected primary plane %u for CRTC %u (pipe %d)\n", plane->plane_id, m_drmCrtc->crtc_id, m_drmCrtcPipe);
        break;
      }

      drmModeFreePlane(plane);
    }

    drmModeFreePlaneResources(planeRes);

    if (!m_drmPlane) {
      fprintf(stderr, "No primary plane found for CRTC %u\n", m_drmCrtc->crtc_id);
      abort();
    }
  }

  // Allocate a dumb framebuffer -- required by the nvidia-drm driver.
  m_drmFb.width = surfaceWidth();
  m_drmFb.height = surfaceHeight();
  m_drmFb.bpp = 32;
  DRM_CHECK(drmIoctl(m_drmFd, DRM_IOCTL_MODE_CREATE_DUMB, &m_drmFb));

  // Add the framebuffer
  uint32_t offset = 0;
  DRM_CHECK(drmModeAddFB2(m_drmFd, surfaceWidth(), surfaceHeight(), DRM_FORMAT_ARGB8888, &m_drmFb.handle, &m_drmFb.pitch, &offset, &m_drmFbBufferId, 0));

  // Atomic modeset: create a mode blob and commit connector->CRTC->plane in one atomic request
  {
    DRM_CHECK(drmModeCreatePropertyBlob(m_drmFd, m_drmModeInfo, sizeof(drmModeModeInfo), &m_drmModeBlobId));

    // Look up required property IDs
    uint32_t connCrtcId   = drmGetPropertyId(m_drmFd, m_drmConnector->connector_id, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID");
    uint32_t crtcActiveId = drmGetPropertyId(m_drmFd, m_drmCrtc->crtc_id, DRM_MODE_OBJECT_CRTC, "ACTIVE");
    uint32_t crtcModeId   = drmGetPropertyId(m_drmFd, m_drmCrtc->crtc_id, DRM_MODE_OBJECT_CRTC, "MODE_ID");
    uint32_t planeFbId    = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "FB_ID");
    uint32_t planeCrtcId  = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_ID");
    uint32_t planeCrtcX   = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_X");
    uint32_t planeCrtcY   = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_Y");
    uint32_t planeCrtcW   = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_W");
    uint32_t planeCrtcH   = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_H");
    uint32_t planeSrcX    = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_X");
    uint32_t planeSrcY    = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_Y");
    uint32_t planeSrcW    = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_W");
    uint32_t planeSrcH    = drmGetPropertyId(m_drmFd, m_drmPlane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_H");

    if (!connCrtcId || !crtcActiveId || !crtcModeId || !planeFbId || !planeCrtcId ||
        !planeCrtcX || !planeCrtcY || !planeCrtcW || !planeCrtcH ||
        !planeSrcX || !planeSrcY || !planeSrcW || !planeSrcH) {
      fprintf(stderr, "Failed to look up one or more DRM atomic properties\n");
      fprintf(stderr, "  connector CRTC_ID=%u  crtc ACTIVE=%u MODE_ID=%u\n", connCrtcId, crtcActiveId, crtcModeId);
      fprintf(stderr, "  plane FB_ID=%u CRTC_ID=%u CRTC_X=%u CRTC_Y=%u CRTC_W=%u CRTC_H=%u\n", planeFbId, planeCrtcId, planeCrtcX, planeCrtcY, planeCrtcW, planeCrtcH);
      fprintf(stderr, "  plane SRC_X=%u SRC_Y=%u SRC_W=%u SRC_H=%u\n", planeSrcX, planeSrcY, planeSrcW, planeSrcH);
      abort();
    }

    drmModeAtomicReq* req = drmModeAtomicAlloc();
    if (!req) {
      fprintf(stderr, "drmModeAtomicAlloc() failed\n");
      abort();
    }

    // Connector: link to CRTC
    drmModeAtomicAddProperty(req, m_drmConnector->connector_id, connCrtcId, m_drmCrtc->crtc_id);

    // CRTC: activate and set mode
    drmModeAtomicAddProperty(req, m_drmCrtc->crtc_id, crtcActiveId, 1);
    drmModeAtomicAddProperty(req, m_drmCrtc->crtc_id, crtcModeId, m_drmModeBlobId);

    // Primary plane: set framebuffer and source/destination rectangles
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeFbId, m_drmFbBufferId);
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeCrtcId, m_drmCrtc->crtc_id);
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeCrtcX, 0);
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeCrtcY, 0);
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeCrtcW, surfaceWidth());
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeCrtcH, surfaceHeight());
    // SRC coords are in 16.16 fixed point
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeSrcX, 0);
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeSrcY, 0);
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeSrcW, (uint64_t)surfaceWidth() << 16);
    drmModeAtomicAddProperty(req, m_drmPlane->plane_id, planeSrcH, (uint64_t)surfaceHeight() << 16);

    int ret = drmModeAtomicCommit(m_drmFd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, nullptr);
    drmModeAtomicFree(req);
    if (ret) {
      fprintf(stderr, "drmModeAtomicCommit() failed: %s (%d)\n", strerror(-ret), ret);
      abort();
    }
    fprintf(stderr, "Atomic modeset commit successful\n");
  }

  // Set up the EGL display
  {
    EGLint attrs[] = {
      EGL_DRM_MASTER_FD_EXT, m_drmFd,
      EGL_NONE
    };
    DRM_CHECK_PTR(m_eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, m_eglDevice, attrs));

    EGLint major, minor;
    EGL_CHECK_BOOL(eglInitialize(m_eglDisplay, &major, &minor));
  }

  // printf("Display Extensions: %s\n\n", eglQueryString(m_eglDisplay, EGL_EXTENSIONS));
  // printf("Device Extensions: %s\n\n", eglQueryDeviceStringEXT(m_eglDevice, EGL_EXTENSIONS));

  CheckExtension("EGL_EXT_output_base");
  CheckExtension("EGL_EXT_output_drm");
  CheckExtension("EGL_EXT_stream_consumer_egloutput");

  // Choose a config and create a context
  EGLint cfg_attr[] = {
    EGL_SURFACE_TYPE, EGL_STREAM_BIT_KHR,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
    EGL_RED_SIZE, 1,
    EGL_GREEN_SIZE, 1,
    EGL_BLUE_SIZE, 1,
    EGL_ALPHA_SIZE, 0,
    EGL_NONE
  };

  int n;
  EGL_CHECK_BOOL(eglChooseConfig(m_eglDisplay, cfg_attr, &m_eglConfig, 1, &n));
  EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  eglBindAPI(EGL_OPENGL_ES_API);
  DRM_CHECK_PTR(m_eglContext = eglCreateContext(m_eglDisplay, m_eglConfig, EGL_NO_CONTEXT, ctx_attr));

  // Activate the display and context to make sure that libepoxy reads the right extension strings.
  // (It uses the display returned by eglGetCurrentDisplay() to detect extensions on first use)
  // If we don't do this, then loading EGL_KHR_stream and EGL_EXT_output_base will fail.
  EGL_CHECK_BOOL(eglMakeCurrent(m_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, m_eglContext));

  // Get the layer for this plane
  EGLAttrib layer_attr[] = {
    EGL_DRM_PLANE_EXT, m_drmPlane->plane_id,
    EGL_NONE
  };

  EGL_CHECK_BOOL(eglGetOutputLayersEXT(m_eglDisplay, layer_attr, &m_eglOutputLayer, 1, &n));

  // Create output stream
  EGLint stream_attr[] = {
    EGL_STREAM_FIFO_LENGTH_KHR, 0, // Mailbox mode
    EGL_NONE};
  DRM_CHECK_PTR(m_eglStream = eglCreateStreamKHR(m_eglDisplay, stream_attr));
  EGL_CHECK_BOOL(eglStreamConsumerOutputEXT(m_eglDisplay, m_eglStream, m_eglOutputLayer));

  // Create surface to feed the stream
  EGLint srf_attr[] = {
    EGL_WIDTH, (EGLint) surfaceWidth(),
    EGL_HEIGHT, (EGLint) surfaceHeight(),
    EGL_NONE};

  DRM_CHECK_PTR(m_eglSurface = eglCreateStreamProducerSurfaceKHR(m_eglDisplay, m_eglConfig, m_eglStream, srf_attr));
  EGL_CHECK_BOOL(eglMakeCurrent(m_eglDisplay, m_eglSurface, m_eglSurface, m_eglContext));

  // Disable vsync, since the camera capture pacing effectively syncs the rendering
  eglSwapInterval(m_eglDisplay, 0);

  m_windowRenderTarget = new RHIEGLSurfaceRenderTargetGL(m_eglDisplay, m_eglSurface);
  m_windowRenderTarget->platformSetUpdatedWindowDimensions(surfaceWidth(), surfaceHeight());
}

RenderBackendDRM::~RenderBackendDRM() {
  eglMakeCurrent(m_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroyContext(m_eglDisplay, m_eglContext);
  eglDestroySurface(m_eglDisplay, m_eglSurface);
  eglDestroyStreamKHR(m_eglDisplay, m_eglStream);

  eglTerminate(m_eglDisplay);

  m_eglDisplay = NULL;
  m_eglSurface = NULL;
  m_eglContext = NULL;
  m_eglStream = NULL;
  // OutputLayer and Device belong to the Display and don't need to be explicitly freed
  m_eglOutputLayer = NULL;
  m_eglDevice = NULL;

  // Remove and destroy DRM framebuffer
  drmModeRmFB(m_drmFd, m_drmFbBufferId);
  m_drmFbBufferId = 0;
  {
    struct drm_mode_destroy_dumb arg;
    memset(&arg, 0, sizeof(drm_mode_destroy_dumb));
    arg.handle = m_drmFb.handle;
    drmIoctl(m_drmFd, DRM_IOCTL_MODE_DESTROY_DUMB, &arg);
  }
  memset(&m_drmFb, 0, sizeof(m_drmFb));

  // Destroy the mode blob
  if (m_drmModeBlobId) {
    drmModeDestroyPropertyBlob(m_drmFd, m_drmModeBlobId);
    m_drmModeBlobId = 0;
  }

  // Free DRM resource allocs
  drmModeFreeConnector(m_drmConnector);
  drmModeFreeEncoder(m_drmEncoder);
  drmModeFreeCrtc(m_drmCrtc);
  drmModeFreePlane(m_drmPlane);
  drmModeFreeResources(m_drmResources);

  m_drmConnector = NULL;
  m_drmEncoder = NULL;
  m_drmCrtc = NULL;
  m_drmPlane = NULL;
  m_drmResources = NULL;

  close(m_drmFd);
  m_drmFd = -1;

}



