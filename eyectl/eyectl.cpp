#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include "effect.h"
#include "effect_runner.h"
#include "noise.h"
#include <glm/gtx/color_space.hpp>

using namespace glm;

bool want_quit = false;
static void signal_handler(int) {
  want_quit = true;

  // Restore signal handlers so the program is still interruptable if clean shutdown gets stuck
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);
}


float randomFloat(float minVal = 0.0f, float maxVal = 1.0f) {
  // drand48() returns a double uniformly distributed over [0.0f, 1.0f)
  return glm::mix(minVal, maxVal, static_cast<float>(drand48()));
}

vec2 randomVec2(vec2 minVal = vec2(0.0f, 0.0f), vec2 maxVal = vec2(1.0f, 1.0f)) {
  return vec2(randomFloat(minVal[0], maxVal[0]), randomFloat(minVal[1], maxVal[1]));
}

class EyeControl : public Effect {
public:
  EyeControl() {}

  vec3 bgColorHSV = vec3(0.0f, 0.8f, 0.75f);
  vec3 pupilColorHSV = vec3(0.1f, 0.55f, 1.0f);

  float intensityScale = 0.5f;

  // rough display center coordinates. units are millimeters
  vec2 center = vec2(80.0f, -20.0f);
  float zLeft = -1.0f, zRight = 1.0f;
  float ledPitch = 3.0f; // millimeters

  vec2 pupilCenter = center;
  vec2 pupilMaxDisplacement = vec2(25.0f, 10.0f);

  float pupilMinSpeed = 100.0f; // mm/sec, for constant speed movement. sets a lower bound
  float pupilMaxMovementTime = 0.125f; // seconds, for constant time movement

  float movementTimeRangeMin = 4.0f;
  float movementTimeRangeMax = 8.0f;

  float blinkTimeRangeMin = 6.0f;
  float blinkTimeRangeMax = 10.0f;
  float blinkDuration = 0.2f; // seconds, full cycle (including dwell time)
  float blinkDwell = 0.05f; // seconds, dwell time at full blink

  // computed state
  vec2 targetPupilCenter = center;
  float currentMovementSpeed = 0;
  float nextMovementTimer = movementTimeRangeMin;
  float nextBlinkTimer = blinkTimeRangeMin;
  float blinkActiveTimer = (blinkDuration - blinkDwell) * 0.5f;
  bool shouldExit = false;

  vec3 modelMin, modelMax; // axis-aligned bounding box, copied from FrameInfo for use in shader

  virtual void beginFrame(const FrameInfo &f) {
    modelMin = f.modelMin;
    modelMax = f.modelMax;
    //const float speed = 10.0;
    //cycle = fmodf(cycle + f.timeDelta * speed, 2 * M_PI);

    nextMovementTimer -= f.timeDelta;
    if (nextMovementTimer < 0) {
      float angle = randomFloat(0.0f, 2.0f * M_PI);
      targetPupilCenter = center + vec2(cosf(angle), sinf(angle)) * randomVec2(vec2(0.0f, 0.0f), pupilMaxDisplacement);

      currentMovementSpeed = glm::length(targetPupilCenter - pupilCenter) / pupilMaxMovementTime; // compute next movement speed using time constant
      currentMovementSpeed = std::max<float>(currentMovementSpeed, pupilMinSpeed); // apply minimum speed for shorter movements
      nextMovementTimer = randomFloat(movementTimeRangeMin, movementTimeRangeMax);
      printf("angle=%fdeg targetPupilCenter=(%f, %f) timer=%f\n", glm::degrees(angle), targetPupilCenter[0], targetPupilCenter[1], nextMovementTimer);
    }

    if (blinkActiveTimer > blinkDuration) { // not blinking
      nextBlinkTimer -= f.timeDelta;
      if (nextBlinkTimer < 0) {
        nextBlinkTimer = randomFloat(blinkTimeRangeMin, blinkTimeRangeMax);
        blinkActiveTimer = 0; // start blink
      }
    } else { // in blink
      blinkActiveTimer += f.timeDelta;
    }

    vec2 distanceToGo = targetPupilCenter - pupilCenter;
    if(glm::length(distanceToGo) > 0.01f) {
      pupilCenter += glm::normalize(distanceToGo) * min<float>(glm::length(distanceToGo), (currentMovementSpeed * f.timeDelta));
    }

    if (want_quit)
      shouldExit = true;
  }

  virtual bool endFrame(const FrameInfo& f) {

    return shouldExit;
  }

  virtual void shader(vec3& rgb, const PixelInfo &p) const {
    if (shouldExit) {
      rgb = vec3(0.0f); // blank display on last frame
      return;
    }

    // Distance from center in XY-plane
    float dist = length(pupilCenter - vec2(p.point));
    float centerMix = smoothstep(16.0f, 30.0f, dist);

    rgb = glm::rgbColor(mix(pupilColorHSV, bgColorHSV, centerMix)) * intensityScale;

    if (blinkActiveTimer < blinkDuration) {
      float phaseLen = ((blinkDuration - blinkDwell) * 0.5f);
      float progress;
      if (blinkActiveTimer < phaseLen) {
        // blinking
        progress = 1.0f - smoothstep(0.0f, phaseLen, blinkActiveTimer);
      } else if (blinkActiveTimer < (phaseLen + blinkDwell)) {
        // dwelling
        progress = 0.0f;
      } else {
        // retracting
        progress = smoothstep(phaseLen + blinkDwell, blinkDuration, blinkActiveTimer);
      }
      float vMin = (p.point.y            - modelMin[1]) / ((modelMax[1] - modelMin[1]) + ledPitch);
      float vMax = (p.point.y + ledPitch - modelMin[1]) / ((modelMax[1] - modelMin[1]) + ledPitch);
      rgb *= smoothstep(vMin, vMax, progress);
    }
  }
};

int main(int argc, char **argv) {
  long int seed;
  getentropy(&seed, sizeof(long int));
  srand48(seed);

  EffectRunner r;

  EyeControl e;
  r.addEffect(&e);

  // Defaults, overridable with command line options
  r.setMaxFrameRate(100);
  r.setLayout("eyectl/layout-eyes.json");

  // Hook signals to stop and blank displays
  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);

  return r.main(argc, argv, /*loop=*/ false);
}

