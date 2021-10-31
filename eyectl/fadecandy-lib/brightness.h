/*
 * Effect that controls the total brightness of another effect,
 * and measures average brightness as well as brightness change rate.
 *
 * Copyright (c) 2014 Micah Elizabeth Scott <micah@scanlime.org>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "effect.h"


class Brightness : public Effect {
public:
    Brightness(Effect &next);

    void set(float averageBrightness);
    void set(float lowerLimit, float upperLimit);

    // Set the gamma value we assume when performing total brightness calculations.
    // Doesn't affect the actual output gamma! We need to sum the brightness in a
    // physically linear space, but we perform the scaling back in our perceptually
    // linear-ish space.
    void setAssumedGamma(float gamma);

    // Gets the last brightness average; will be roughly between lowerLimit and upperLimit.
    float getAverageBrightness() const;

    // Get the total squared brightness delta between the last two frames
    float getTotalBrightnessDelta() const;

    virtual void beginFrame(const FrameInfo& f);
    virtual bool endFrame(const FrameInfo& f);
    virtual void debug(const DebugInfo& f);
    virtual void shader(glm::vec3& rgb, const PixelInfo& p) const;

private:
    Effect &next;
    float lowerLimit, upperLimit;
    float currentScale;
    float latestAverage;
    float totalBrightnessDelta;
    unsigned numIters;

    std::vector<glm::vec3> *prevColors;
    std::vector<glm::vec3> *nextColors;

    std::vector<glm::vec3> colorBuffer[2];

    static const unsigned gammaTableSize = 256;
    float gammaTable[gammaTableSize];
    float gamma;
};


/*****************************************************************************************
 *                                   Implementation
 *****************************************************************************************/


inline Brightness::Brightness(Effect &next)
    : next(next),
      lowerLimit(0),
      upperLimit(1),
      currentScale(1),
      latestAverage(0),
      totalBrightnessDelta(0),
      numIters(0)
{
    // Fadecandy default
    setAssumedGamma(2.5);

    prevColors = &colorBuffer[0];
    nextColors = &colorBuffer[1];
}

inline void Brightness::set(float averageBrightness)
{
    lowerLimit = upperLimit = averageBrightness;
}

inline void Brightness::set(float lowerLimit, float upperLimit)
{
    this->lowerLimit = lowerLimit;
    this->upperLimit = upperLimit;
}

inline void Brightness::setAssumedGamma(float gamma)
{
    this->gamma = gamma;
    for (unsigned i = 0; i < gammaTableSize; i++) {
        gammaTable[i] = powf(i / float(gammaTableSize + 1), gamma);
    }
}

inline float Brightness::getAverageBrightness() const
{
    return latestAverage;
}

inline float Brightness::getTotalBrightnessDelta() const
{
    return totalBrightnessDelta;
}

inline void Brightness::beginFrame(const FrameInfo& f)
{
    next.beginFrame(f);
    std::swap(nextColors, prevColors);

    if (colorBuffer[0].size() != f.pixels.size()) {
        for (unsigned i = 0; i < 2; i++) {
           colorBuffer[i].resize(f.pixels.size());
           std::fill(colorBuffer[i].begin(), colorBuffer[i].end(), glm::vec3(0,0,0));
        }
    }

    unsigned count = 0;
    float deltaAccumulator = 0;

    // Calculate the next effect's pixels, storing them all. Also count the total number
    // of mapped pixels, ignoring any unmapped ones.

    {
        PixelInfoIter pi = f.pixels.begin();
        PixelInfoIter pe = f.pixels.end();
        std::vector<glm::vec3>::iterator nci = nextColors->begin();
        std::vector<glm::vec3>::iterator pci = prevColors->begin();

        for (;pi != pe; ++pi, ++nci, ++pci) {
            if (pi->isMapped()) {
                glm::vec3 rgb(0, 0, 0);
                next.shader(rgb, *pi);
                next.postProcess(rgb, *pi);
                count++;

                *nci = rgb;
                deltaAccumulator += sqrlen(rgb - *pci);
            }
        }
    }

    const float deltaAccumulatorFilterRate = 0.05;
    totalBrightnessDelta += (deltaAccumulator - totalBrightnessDelta) * deltaAccumulatorFilterRate;

    if (!(totalBrightnessDelta >= 0)) {
        // NaN or negative
        totalBrightnessDelta = 0;
    }

    if (count == 0) {
        // No LEDs mapped
        return;
    }

    // Iterative algorithm to adjust brightness scaling. I'm not sure a closed-form
    // solution exists- this is complicated for multiple reasons. We want to scale the
    // entire image in a perceptually linear way, but the final brightness we're interested
    // in is related to the total linear intensity of all LEDs. Additionally, the brightness
    // is clamped at each LED, so we may need to increase the brightness of other LEDs to
    // compensate for individual LEDs that can't get any brighter. Usually this only takes
    // a few iterations to converge.

    const unsigned maxIters = 50;
    const float epsilon = 1e-3;

    unsigned iter = 0;
    float avg;
    float scale = 1.0;

    for (; iter < maxIters; iter++) {

        std::vector<glm::vec3>::iterator ci = nextColors->begin();
        std::vector<glm::vec3>::iterator ce = nextColors->end();
        PixelInfoIter pi = f.pixels.begin();
        avg = 0;

        for (;ci != ce; ++ci, ++pi) {
            glm::vec3& rgb = *ci;

            // Simulated linear brightness, using current scale
            if (pi->isMapped()) {
                for (unsigned i = 0; i < 3; i++) {
                    float c = rgb[i] * scale;
                    avg += gammaTable[std::max<int>(0, std::min<int>(gammaTableSize - 1, c * float(gammaTableSize - 1)))];
                }
            }
        }

        avg /= count;

        float adjustment;
        if (avg < lowerLimit) {
            // Make brighter, operate against the lower limit
            adjustment = powf(lowerLimit / avg, 1.0f / gamma);
        } else if (avg > upperLimit) {
            // Make dimmer, operate against the upper limit
            adjustment = powf(upperLimit / avg, 1.0f / gamma);
        } else {
            // Not hitting any limits
            break;
        }

        scale = std::max(epsilon, scale * adjustment);

        // Was this adjustment negligible? We can quit early.
        if (fabsf(adjustment - 1.0f) < epsilon) {
            break;
        }

    }

    numIters = iter;
    currentScale = scale;
    latestAverage = avg;
}

inline bool Brightness::endFrame(const FrameInfo& f)
{
    next.endFrame(f);
    return Effect::endFrame(f);
}

inline void Brightness::debug(const DebugInfo& d)
{
    next.debug(d);
    fprintf(stderr, "\t[brightness] limits = [%f, %f]\n", lowerLimit, upperLimit);
    fprintf(stderr, "\t[brightness] currentScale = %f\n", currentScale);
    fprintf(stderr, "\t[brightness] latestAverage = %f\n", latestAverage);
    fprintf(stderr, "\t[brightness] totalBrightnessDelta = %f\n", totalBrightnessDelta);
    fprintf(stderr, "\t[brightness] iterations = %d\n", numIters);
}

inline void Brightness::shader(glm::vec3& rgb, const PixelInfo& p) const
{
    rgb = (*nextColors)[p.index] * currentScale;
}
