/*
 * LED Effect mixing board: Runs any number of effects concurrently,
 * and mix the results together.
 *
 * This is an optional layer. You can connect an Effect directly to
 * the EffectRunner, and this skips a lot of complexity and memory
 * usage. But if you add an EffectMixer, we use multiple threads and
 * we keep a separate RGB buffer for each effect. This allows single
 * effects or multiple effects to be sliced over multiple CPU cores.
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

#include <queue>
#include <vector>

#include "effect.h"
#include "tinythread.h"


class EffectMixer : public Effect {
public:
    EffectMixer();
    ~EffectMixer();

    // Managing channels
    int numChannels();
    void clear();
    void set(Effect *effect);
    int add(Effect *effect, float fader = 1.0);
    int find(Effect *effect);
    void remove(int index);
    void remove(Effect *effect);
    void setFader(int channel, float fader);
    void setFader(Effect *effect, float fader);

    // Set number of threads. By default, we auto-detect
    void setConcurrency(unsigned numThreads);

    virtual void shader(glm::vec3& rgb, const PixelInfo& p) const;
    virtual void postProcess(const glm::vec3& rgb, const PixelInfo& p);
    virtual void beginFrame(const FrameInfo& f);
    virtual bool endFrame(const FrameInfo& f);
    virtual void debug(const DebugInfo& d);

private:
    struct Channel {
        Effect *effect;
        float fader;
        std::vector<glm::vec3> colors;
    };

    struct Task {
        Channel *channel;
        const Effect::PixelInfo *pixelInfo;
        unsigned begin;
        unsigned end;
    };

    struct ThreadContext {
        EffectMixer *mixer;
        tthread::thread *thread;
        bool runFlag;
    };

    // Channels only to be modified when threads are idle
    std::vector<Channel> channels;

    // Running threads
    std::vector<ThreadContext*> threads;
    unsigned numThreadsConfigured;

    // Lock rank: Acquire taskLock prior to completeLock.

    // Task queue
    tthread::mutex taskLock;
    tthread::condition_variable taskCond;
    std::queue<Task> tasks;

    // Completion status
    tthread::mutex completeLock;
    tthread::condition_variable completeCond;
    unsigned pendingTasks;

    void changeNumberOfThreads(unsigned count);
    static void threadFunc(void *context);
    void worker(ThreadContext &context);
};


/*****************************************************************************************
 *                                   Implementation
 *****************************************************************************************/


inline EffectMixer::EffectMixer()
    : numThreadsConfigured(0)   // Auto-detect
{}

inline EffectMixer::~EffectMixer()
{
    changeNumberOfThreads(0);
}

inline void EffectMixer::setConcurrency(unsigned numThreads)
{
    // Threads created/destroyed lazily
    numThreadsConfigured = numThreads;
}

inline int EffectMixer::numChannels()
{
    return channels.size();
}

inline int EffectMixer::add(Effect *effect, float fader)
{
    Channel c;

    c.effect = effect;
    c.fader = fader;

    int index = channels.size();
    channels.push_back(c);
    return index;
}

inline void EffectMixer::clear()
{
    channels.clear();
}

inline void EffectMixer::set(Effect *effect)
{
    clear();
    add(effect);
}

inline int EffectMixer::find(Effect *effect)
{
    for (unsigned i = 0; i < channels.size(); i++) {
        if (channels[i].effect == effect) {
            return i;
        }
    }
    return -1;
}

inline void EffectMixer::remove(int index)
{
    if (index >= 0 && index < (int)channels.size()) {
        channels.erase(channels.begin() + index);
    }
}

inline void EffectMixer::remove(Effect *effect)
{
    remove(find(effect));
}

inline void EffectMixer::setFader(int channel, float fader)
{
    if (channel >= 0 && channel < (int)channels.size()) {
        channels[channel].fader = fader;
    }
}

inline void EffectMixer::setFader(Effect *effect, float fader)
{
    setFader(find(effect), fader);
}

inline void EffectMixer::shader(glm::vec3& rgb, const PixelInfo& p) const
{
    // Mix together results from channel buffers.
    // Assumes the channel's color buffer has already been set up and sized by beginFrame().

    glm::vec3 total(0,0,0);

    for (std::vector<Channel>::const_iterator i = channels.begin(), e = channels.end(); i != e; ++i) {
        float f = i->fader;
        if (f) {
            total += i->colors[p.index] * f;
        }
    }

    rgb = total;
}

inline void EffectMixer::postProcess(const glm::vec3& rgb, const PixelInfo& p)
{
    // Allow all channels to post-process their result, without parallelism.

    for (std::vector<Channel>::iterator i = channels.begin(), e = channels.end(); i != e; ++i) {
        Channel &c = *i;
        float f = c.fader;
        if (f) {
            c.effect->postProcess(c.colors[p.index], p);
        }
    }
}

inline bool EffectMixer::endFrame(const FrameInfo& f)
{
    bool lastFrame = false;
    for (unsigned i = 0; i < channels.size(); ++i) {
        lastFrame |= channels[i].effect->endFrame(f);
    }
    return lastFrame | Effect::endFrame(f);
}

inline void EffectMixer::debug(const DebugInfo& d)
{
    for (unsigned i = 0; i < channels.size(); ++i) {
        channels[i].effect->debug(d);
    }
}

inline void EffectMixer::changeNumberOfThreads(unsigned count)
{
    while (threads.size() < numThreadsConfigured) {
        // Create thread
        ThreadContext *tc = new ThreadContext;
        tc->mixer = this;
        tc->runFlag = true;
        tc->thread = new tthread::thread(threadFunc, tc);
        threads.push_back(tc);
    }

    while (threads.size() > numThreadsConfigured) {
        // Signal a thread to stop
        ThreadContext *tc = threads.back();
        threads.pop_back();

        tc->runFlag = false;
        taskCond.notify_all();

        tc->thread->join();
        delete tc->thread;
        delete tc;
    }
}

inline void EffectMixer::beginFrame(const FrameInfo& f)
{
    // Auto-detect thread count
    if (numThreadsConfigured == 0) {
        numThreadsConfigured = tthread::thread::hardware_concurrency();
    }

    // Create/destroy threads, to reach the requested pool size
    changeNumberOfThreads(numThreadsConfigured);

    /*
     * Setup for each effect:
     *   - Send a beginFrame() message
     *   - Keep track of the total pixel count, for sizing our batches
     *   - Size our channel's color buffer
     */

    unsigned totalPixels = 0;
    unsigned modelPixels = f.pixels.size();

    for (unsigned i = 0; i < channels.size(); ++i) {
        Channel &c = channels[i];

        c.effect->beginFrame(f);
        c.colors.resize(modelPixels);
        if (c.fader) {
            totalPixels += modelPixels;
        }
    }

    // Try to size the batches so we give each CPU a few tasks, so that if our
    // workload is asymmetric we'll end up with room to rebalance.

    unsigned batchSize = 1 + totalPixels / (numThreadsConfigured * 3);

    // Create tasks for each active effect, and wait for our thread pool to process them.
    // Note our lock ranking requirements: taskLock acquired before completeLock.

    taskLock.lock();
    unsigned numTasks = 0;

    for (unsigned i = 0; i < channels.size(); ++i) {
        Channel &c = channels[i];
        if (c.fader) {
            Task t;
            t.channel = &c;
            t.pixelInfo = &f.pixels[0];
            t.begin = 0;

            while (t.begin < modelPixels) {
                t.end = std::min<unsigned>(modelPixels, t.begin + batchSize);
                tasks.push(t);
                t.begin = t.end;
                numTasks++;
            }
        }
    }

    completeLock.lock();
    pendingTasks = numTasks;
    taskCond.notify_all();
    taskLock.unlock();

    while (pendingTasks) {
        completeCond.wait(completeLock);
    }

    completeLock.unlock();
}

inline void EffectMixer::threadFunc(void *context)
{
    ThreadContext* c = (ThreadContext*) context;
    c->mixer->worker(*c);
}

inline void EffectMixer::worker(ThreadContext &context)
{
    while (true) {
        Task currentTask;

        // Dequeue a task
        taskLock.lock();
        while (tasks.empty()) {
            if (!context.runFlag) {
                // Thread exiting
                return;
            }
            taskCond.wait(taskLock);
        }
        currentTask = tasks.front();
        tasks.pop();
        taskLock.unlock();

        // Process a block of pixels

        Channel &c = *currentTask.channel;
        Effect *effect = c.effect;

        for (unsigned i = currentTask.begin; i != currentTask.end; ++i) {
            const Effect::PixelInfo &p = currentTask.pixelInfo[i];
            if (p.isMapped()) {
                glm::vec3 color(0, 0, 0);
                effect->shader(color, p);
                c.colors[i] = color;
            }
        }

        // Completion notification
        completeLock.lock();
        pendingTasks--;
        completeCond.notify_all();
        completeLock.unlock();
    }
}
