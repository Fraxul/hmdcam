#include "common/FxRenderView.h"
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

void FxRenderView::recomputeFrustumData() {
  viewProjectionMatrix = projectionMatrix * viewMatrix;
  //frustum = FxFrustum(viewProjectionMatrix);
}

