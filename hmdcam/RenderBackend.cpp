#include "RenderBackend.h"
#include <stdio.h>
#include <string.h>

ERenderBackend renderBackendStringToEnum(const char* s) {
  
  if (!strcasecmp(s, "drm")) {
    return kRenderBackendDRM;
  } else if (!strcasecmp(s, "wayland")) {
    return kRenderBackendWayland;
  } else if ((!strcasecmp(s, "vkdirect")) || (!strcasecmp(s, "vulkan"))) {
    return kRenderBackendVKDirect;
  } else if ((!strcasecmp(s, "surfaceless")) || (!strcasecmp(s, "null"))) {
    return kRenderBackendSurfaceless;
  } else {
    fprintf(stderr, "renderBackendStringToEnum: unrecognized backend name \"%s\"\n", s);
    return kRenderBackendNone;
  }
}

extern RenderBackend* createWaylandBackend();
extern RenderBackend* createDRMBackend();
extern RenderBackend* createVKDirectBackend();
extern RenderBackend* createSurfacelessBackend();

/*static*/ RenderBackend* RenderBackend::create(ERenderBackend rb) {
  switch (rb) {
    case kRenderBackendDRM: return createDRMBackend();
    case kRenderBackendWayland: return createWaylandBackend();
    case kRenderBackendVKDirect: return createVKDirectBackend();
    case kRenderBackendSurfaceless: return createSurfacelessBackend();

    default:
    case kRenderBackendNone: return NULL;
  };
}

