/*
 * LED Effect Framework
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

#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <sys/time.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "effect.h"
#include "opc_client.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/filestream.h"
#include "rapidjson/document.h"


class EffectRunner {
public:
    EffectRunner();

    bool setServer(const char *hostport);
    bool setLayout(const char *filename);
    void setEffect(Effect* effect);
    void addEffect(Effect* effect);
    void setMaxFrameRate(float fps);
    void setVerbose(bool verbose = true);

    bool hasLayout() const;
    const rapidjson::Document& getLayout() const;
    Effect* getEffect() const;
    bool isVerbose() const;
    OPCClient& getClient();

    // Access to most recent framebuffer information
    const Effect::PixelInfoVec& getPixelInfo() const;
    const uint8_t* getPixel(unsigned index) const;
    void getPixelColor(unsigned index, glm::vec3 &rgb) const;

    // Time stats
    float getFrameRate() const;
    float getTimePerFrame() const;
    float getBusyTimePerFrame() const;
    float getIdleTimePerFrame() const;
    float getPercentBusy() const;

    struct FrameStatus {
        float timeDelta;
        bool debugOutput;
        bool lastFrame;
    };

    // Main loop body
    FrameStatus doFrame();
    FrameStatus doFrame(float timeDelta);

    // Minimal main loop
    void run();

    // Simple argument parsing and optional main loop
    bool parseArguments(int argc, char **argv);
    int main(int argc, char **argv, bool loop = true);

protected:
    // Extensibility for argument parsing
    virtual bool parseArgument(int &i, int &argc, char **argv);
    virtual bool validateArguments();
    virtual void argumentUsage();

private:
    OPCClient opc;
    rapidjson::Document layout;
    Effect *effect;
    std::vector<Effect*> effects;
    std::vector<uint8_t> frameBuffer;
    Effect::FrameInfo frameInfo;

    float minTimeDelta;
    float currentDelay;
    float filteredTimeDelta;
    float debugTimer;
    float speed;
    bool verbose;
    struct timeval lastTime;
    float jitterStatsMin;
    float jitterStatsMax;

    void usage(const char *name);
    void debug();
};


/*****************************************************************************************
 *                                   Implementation
 *****************************************************************************************/


inline EffectRunner::EffectRunner()
    : opc(),
      layout(),
      effect(0),
      effects(),
      frameBuffer(),
      frameInfo(),
      minTimeDelta(0),
      currentDelay(0),
      filteredTimeDelta(0),
      debugTimer(0),
      speed(1.0),
      verbose(false),
      jitterStatsMin(1),
      jitterStatsMax(0)
{
    lastTime.tv_sec = 0;
    lastTime.tv_usec = 0;

    // Defaults
    setMaxFrameRate(300);
    setServer("localhost");
}

inline void EffectRunner::setMaxFrameRate(float fps)
{
    minTimeDelta = 1.0 / fps;
}

inline void EffectRunner::setVerbose(bool verbose)
{
    this->verbose = verbose;
}

inline bool EffectRunner::setServer(const char *hostport)
{
    return opc.resolve(hostport);
}

inline bool EffectRunner::setLayout(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        return false;
    }

    rapidjson::FileStream istr(f);
    layout.ParseStream<0>(istr);
    fclose(f);

    if (layout.HasParseError()) {
        return false;
    }
    if (!layout.IsArray()) {
        return false;
    }

    // Set up an empty framebuffer, with OPC packet header
    int frameBytes = layout.Size() * 3;
    frameBuffer.resize(sizeof(OPCClient::Header) + frameBytes);
    OPCClient::Header::view(frameBuffer).init(0, opc.SET_PIXEL_COLORS, frameBytes);

    // Init pixel info
    frameInfo.init(layout);

    return true;
}

inline const rapidjson::Document& EffectRunner::getLayout() const
{
    return layout;
}

inline bool EffectRunner::hasLayout() const
{
    return layout.IsArray();
}

inline void EffectRunner::setEffect(Effect *effect)
{
    effects.clear();
    effects.push_back( effect );
    this->effect = effect;
}

inline void EffectRunner::addEffect( Effect *effect )
{
    effects.push_back( effect );
}

inline Effect* EffectRunner::getEffect() const
{
    return effect;
}

inline bool EffectRunner::isVerbose() const
{
    return verbose;
}

inline float EffectRunner::getFrameRate() const
{
    return filteredTimeDelta > 0.0f ? 1.0f / filteredTimeDelta : 0.0f;
}

inline float EffectRunner::getTimePerFrame() const
{
    return filteredTimeDelta;
}

inline float EffectRunner::getBusyTimePerFrame() const
{
    return getTimePerFrame() - getIdleTimePerFrame();
}

inline float EffectRunner::getIdleTimePerFrame() const
{
    return std::max(0.0f, currentDelay);
}

inline float EffectRunner::getPercentBusy() const
{
    return 100.0f * getBusyTimePerFrame() / getTimePerFrame();
}

inline void EffectRunner::run()
{
    FrameStatus status;
    do
    {
        status = doFrame();
    }
    while( !status.lastFrame );
}
   
inline EffectRunner::FrameStatus EffectRunner::doFrame()
{
    struct timeval now;

    gettimeofday(&now, 0);
    float delta = (now.tv_sec - lastTime.tv_sec)
        + 1e-6 * (now.tv_usec - lastTime.tv_usec);
    lastTime = now;

    // Max timestep; jump ahead if we get too far behind.
    const float maxStep = 0.1;
    if (delta > maxStep) {
        delta = maxStep;
    }

    return doFrame(delta);
}

inline EffectRunner::FrameStatus EffectRunner::doFrame(float timeDelta)
{
    FrameStatus frameStatus;

    // Effects may get a modified view of time
    frameStatus.timeDelta   = frameInfo.timeDelta = timeDelta * speed;
    frameStatus.debugOutput = false;
    frameStatus.lastFrame   = false;

    jitterStatsMin = std::min(jitterStatsMin, frameStatus.timeDelta);
    jitterStatsMax = std::max(jitterStatsMax, frameStatus.timeDelta);

    if (getEffect() && hasLayout()) {
        effect->beginFrame(frameInfo);

        // Only calculate the effect if we have a connection
        if (opc.tryConnect()) {

            uint8_t *dest = OPCClient::Header::view(frameBuffer).data();

            for (Effect::PixelInfoIter i = frameInfo.pixels.begin(), e = frameInfo.pixels.end(); i != e; ++i) {
                glm::vec3 rgb(0, 0, 0);
                const Effect::PixelInfo &p = *i;

                if (p.isMapped()) {
                    effect->shader(rgb, p);
                    effect->postProcess(rgb, p);
                }

                for (unsigned i = 0; i < 3; i++) {
                    *(dest++) = std::min<int>(255, std::max<int>(0, rgb[i] * 255 + 0.5));
                }
            }

            opc.write(frameBuffer);
        }

        frameStatus.lastFrame = effect->endFrame(frameInfo);
    }

    // Low-pass filter for timeDelta, to estimate our frame rate
    const float filterGain = 0.05;
    filteredTimeDelta += (timeDelta - filteredTimeDelta) * filterGain;

    // Negative feedback loop to adjust the delay until we hit a target frame rate.
    // This lets us hit the target rate smoothly, without a lot of jitter between frames.
    // If we calculated a new delay value on each frame, we'd easily end up alternating
    // between too-long and too-short frame delays.
    currentDelay += (minTimeDelta - timeDelta) * filterGain;

    // Make sure filteredTimeDelta >= currentDelay. (The "busy time" estimate will be >= 0)
    filteredTimeDelta = std::max(filteredTimeDelta, currentDelay);

    // Periodically output debug info, if we're in verbose mode
    if (verbose) {
        const float debugInterval = 1.0f;
        if ((debugTimer += timeDelta) > debugInterval) {
            debugTimer = fmodf(debugTimer, debugInterval);
            frameStatus.debugOutput = true;
            debug();
        }
    }

    // Add the extra delay, if we have one. This is how we throttle down the frame rate.
    if (currentDelay > 0) {
        usleep(currentDelay * 1e6);
    }

    return frameStatus;
}

inline OPCClient& EffectRunner::getClient()
{
    return opc;
}

inline const Effect::PixelInfoVec& EffectRunner::getPixelInfo() const
{
    return frameInfo.pixels;
}

inline const uint8_t* EffectRunner::getPixel(unsigned index) const
{
    return OPCClient::Header::view(frameBuffer).data() + index * 3;
}

inline void EffectRunner::getPixelColor(unsigned index, glm::vec3 &rgb) const
{
    const uint8_t *byte = getPixel(index);
    for (unsigned i = 0; i < 3; i++) {
        rgb[i] = *(byte++) / 255.0f;
    }
}

inline bool EffectRunner::parseArguments(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (!parseArgument(i, argc, argv)) {
            usage(argv[0]);
            return false;
        }
    }

    if (!validateArguments()) {
        usage(argv[0]);
        return false;
    }

    return true;
}

inline int EffectRunner::main(int argc, char **argv, bool loop)
{
    if (!parseArguments(argc, argv)) {
        return 1;
    }

    do
    {
       for( std::vector<Effect*>::iterator i( effects.begin() );
            i != effects.end(); ++i )
       {
          effect = *i;
          run();
       }
    }
    while( loop );
    return 0;
}

inline void EffectRunner::usage(const char *name)
{
    fprintf(stderr, "usage: %s ", name);
    argumentUsage();
    fprintf(stderr, "\n");
}

inline void EffectRunner::debug()
{
    fprintf(stderr, " %7.2f FPS -- %6.2f%% CPU [%.3fms busy, %.3fms idle, %.3fms jitter]\n",
        getFrameRate(),
        getPercentBusy(),
        1e3f * getBusyTimePerFrame(),
        1e3f * getIdleTimePerFrame(),
        1e3f * (jitterStatsMax - jitterStatsMin));

    jitterStatsMax = 0;
    jitterStatsMin = 1e10;

    if (effect) {
        Effect::DebugInfo d(*this);
        effect->debug(d);
    }
}

inline bool EffectRunner::parseArgument(int &i, int &argc, char **argv)
{
    if (!strcmp(argv[i], "-v")) {
        setVerbose();
        return true;
    }

    if (!strcmp(argv[i], "-fps") && (i+1 < argc)) {
        float rate = atof(argv[++i]);
        if (rate <= 0) {
            fprintf(stderr, "Invalid frame rate\n");
            return false;
        }
        setMaxFrameRate(rate);
        return true;
    }

    if (!strcmp(argv[i], "-speed") && (i+1 < argc)) {
        speed = atof(argv[++i]);
        if (speed <= 0) {
            fprintf(stderr, "Invalid speed\n");
            return false;
        }
        return true;
    }

    if (!strcmp(argv[i], "-layout") && (i+1 < argc)) {
        if (!setLayout(argv[++i])) {
            fprintf(stderr, "Can't load layout from %s\n", argv[i]);
            return false;
        }
        return true;
    }

    if (!strcmp(argv[i], "-server") && (i+1 < argc)) {
        if (!setServer(argv[++i])) {
            fprintf(stderr, "Can't resolve server name %s\n", argv[i]);
            return false;
        }
        return true;
    }

    return false;
}

inline bool EffectRunner::validateArguments()
{
    if (!hasLayout()) {
        fprintf(stderr, "No layout specified\n");
        return false;
    }

    return true;
}

inline void EffectRunner::argumentUsage()
{
    fprintf(stderr, "[-v] [-fps LIMIT] [-speed MULTIPLIER] [-layout FILE.json] [-server HOST[:port]]");
}
