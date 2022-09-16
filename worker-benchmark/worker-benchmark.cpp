#include <stdio.h>
#include <libgen.h>
#include <unistd.h>
#include <sys/wait.h>
#include "common/SHMSegment.h"
#include "common/DepthMapSHM.h"
#include "common/Timing.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/mean.hpp>

typedef boost::accumulators::accumulator_set<float, boost::accumulators::stats<
        boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::mean,
        boost::accumulators::tag::median
      > > timingStatsAccumulator_t;

void printTimingStats(timingStatsAccumulator_t& accum, bool reset = true) {
  printf("min=%.3gms max=%.3gms mean=%.3gms median=%.3gms\n",
    boost::accumulators::min(accum),
    boost::accumulators::max(accum),
    boost::accumulators::mean(accum),
    boost::accumulators::median(accum));
  if (reset)
    accum = {};
}


inline uint32_t roundUp4K(uint32_t x) { return (x + 4095) & (~(4095U)); }

void pitchCopy(void* outBase, size_t outPitchBytes, void* inBase, size_t inPitchBytes, size_t rowLengthBytes, size_t height) {
  assert(rowLengthBytes <= outPitchBytes);
  assert(rowLengthBytes <= inPitchBytes);
  for (size_t y = 0; y < height; ++y) {
    memcpy(reinterpret_cast<char*>(outBase) + (outPitchBytes * y), reinterpret_cast<const char*>(inBase) + (inPitchBytes * y), rowLengthBytes);
  }
}

enum DepthMapGeneratorBackend {
  kDepthBackendNone,
  kDepthBackendDGPU,
  kDepthBackendDepthAI,
  //kDepthBackendVPI
};

DepthMapGeneratorBackend depthBackendStringToEnum(const char* backendStr) {
  if (!strcasecmp(backendStr, "none")) {
    return kDepthBackendNone;
  } else if ((!strcasecmp(backendStr, "dgpu")) || (!strcasecmp(backendStr, "cuda"))) {
    return kDepthBackendDGPU;
  } else if ((!strcasecmp(backendStr, "depthai")) || (!strcasecmp(backendStr, "depth-ai"))) {
    return kDepthBackendDepthAI;
//  } else if ((!strcasecmp(backendStr, "vpi")) || (!strcasecmp(backendStr, "vpi2"))) {
//    return kDepthBackendVPI;
  } else {
    fprintf(stderr, "depthBackendStringToEnum: unrecognized worker type \"%s\"\n", backendStr);
    return kDepthBackendNone;
  }
}

int spawnDepthWorker(DepthMapGeneratorBackend backend) {
  char* exepath = realpath("/proc/self/exe", NULL);
  assert(exepath);

  std::string workerBin = std::string(dirname(exepath));
  free(exepath);

  switch (backend) {
    case kDepthBackendNone:
      assert(false && "spawnDepthWorker: can't spawn process for backend type kDepthBackendNone");
      break;

    case kDepthBackendDGPU:
      workerBin += "/dgpu-worker";
      break;

    case kDepthBackendDepthAI:
      workerBin += "/depthai-worker";
      break;

    default:
      assert(false && "spawnDepthWorker: invalid backend enum");
  };

  int pid = vfork();
  if (!pid) {
    // spawn child process
    char* argv0 = const_cast<char*>(workerBin.c_str());
    char* args[] = { const_cast<char*>(argv0), NULL };
    if (-1 == execv(argv0, args)) {
      printf("execv() failed: %s\n", strerror(errno));
      _exit(-1);
    }
  }

  printf("Spawning Depth worker binary %s as PID %d\n", workerBin.c_str(), pid);
  return pid;
}

void waitForDepthWorkerReady(int pid, sem_t* sem, unsigned int timeout_sec) {
  char err[128];

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  for (unsigned int i = 0; i < timeout_sec; ++i) {
    ts.tv_sec += 1;
    int res = sem_timedwait(sem, &ts);
    if (res == 0)
      return; // OK
    else if (errno != ETIMEDOUT) {
      sprintf(err, "waitForDepthWorkerReady(): sem_timedwait: %s", strerror(errno));
      throw std::runtime_error(err);
    }
    int wstatus;
    if (waitpid(pid, &wstatus, WNOHANG) > 0) {
      if (WIFEXITED(wstatus)) {
        sprintf(err, "Depth worker exited; status %d", WEXITSTATUS(wstatus));
      } else if (WIFSIGNALED(wstatus)) {
        sprintf(err, "Depth worker exited; signal %d", WTERMSIG(wstatus));
      } else {
        sprintf(err, "Depth worker exited; unknown reason");
      }
      throw std::runtime_error(err);
    }

  }
  throw std::runtime_error("Timed out waiting for Depth worker to initialize");
}

int main(int argc, char* argv[]) {
  DepthMapGeneratorBackend depthBackend = kDepthBackendNone;
  int viewCount = 1;
  int resX = 0, resY = 0;
  uint32_t endFrameCount = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth-backend")) {
      if (i == (argc - 1)) {
        printf("--depth-backend: requires argument\n");
        return 1;
      }
      depthBackend = depthBackendStringToEnum(argv[++i]);
    } else if (!strcmp(argv[i], "--frameCount")) {
      if (i == (argc - 1)) {
        printf("--frameCount: requires argument: count\n");
        return 1;
      }
      endFrameCount = atoi(argv[++i]);
      if (viewCount <= 1) {
        printf("--frameCount: invalid argument\n");
        return 1;
      }
    } else if (!strcmp(argv[i], "--views")) {
      if (i == (argc - 1)) {
        printf("--views: requires argument: count\n");
        return 1;
      }
      viewCount = atoi(argv[++i]);
      if (viewCount <= 1) {
        printf("--views: invalid argument\n");
        return 1;
      }
    } else if (!strcmp(argv[i], "--res")) {
      if (i >= (argc - 2)) {
        printf("--res: requires 2 arguments: width height\n");
        return 1;
      }
      resX = atoi(argv[++i]);
      resY = atoi(argv[++i]);
      if (!(resX && resY)) {
        printf("--res: invalid arguments\n");
        return 1;
      }
    } else {
      printf("Unrecognized argument %s\n", argv[i]);
      return 1;
    }
  }


  auto shm = SHMSegment<DepthMapSHM>::createSegment("depth-worker", 16*1024*1024);
  auto segment = shm->segment();

  size_t lastOffset = 0;

  segment->m_activeViewCount = viewCount;
  for (size_t viewIdx = 0; viewIdx < viewCount; ++viewIdx) {
    DepthMapSHM::ViewParams& vp = segment->m_viewParams[viewIdx];

    for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
      // Read input images
      char filename[32];
      sprintf(filename, "view%zu_%s.png", viewIdx, eyeIdx == 0 ? "left" : "right");
      cv::Mat m;
      try {
        m = cv::imread(filename, cv::IMREAD_GRAYSCALE);
      } catch (const std::exception& ex) {
        printf("Error opening %s: %s\n", filename, ex.what());
        abort();
      }

      assert(m.type() == CV_8UC1);

      if (resX && resY) {
        cv::resize(m, m, cv::Size(resX, resY));
      }
      if (eyeIdx == 0) {
        vp.width = m.cols;
        vp.height = m.rows;
      } else {
        // ensure L and R dims are identical
        assert(vp.width == m.cols);
        assert(vp.height == m.rows);
      }

      // Allocate SHM area for inputs
      vp.inputOffset[eyeIdx] = lastOffset;
      vp.inputPitchBytes = vp.width;

      lastOffset += roundUp4K(vp.inputPitchBytes * vp.height);

      // Copy input images to SHM data area
      pitchCopy(segment->data() + vp.inputOffset[eyeIdx], vp.inputPitchBytes, m.data, m.step, vp.width, vp.height);
    }

    // Allocate SHM area for output
    vp.outputOffset = lastOffset;
    vp.outputPitchBytes = vp.width * 2; // should work for either 8 or 16 bit disparity
    lastOffset += roundUp4K(vp.outputPitchBytes * vp.height);
  }

  if (getenv("DEPTH_WORKER_ATTACH")) {
    // Debug support for launching the worker process externally (through a debugger or NSight)
    printf("Waiting for externally-spawned depth worker...\n");
    int res = sem_wait(&segment->m_workerReadySem);
    if (res != 0) {
      perror("sem_wait()");
      abort();
    }
  } else {
    int timeout_sec = 5;
    {
      const char* timeoutEnvStr = getenv("DEPTH_WORKER_TIMEOUT");
      if (timeoutEnvStr) {
        int timeoutEnv = atoi(timeoutEnvStr);
        if (timeoutEnv <= 0) {
          timeout_sec = UINT_MAX;
          fprintf(stderr, "DepthMapGeneratorSHM: DEPTH_WORKER_TIMEOUT <= 0, waiting forever\n");
        } else {
          timeout_sec = timeoutEnv;
          fprintf(stderr, "DepthMapGeneratorSHM: DEPTH_WORKER_TIMEOUT is %u seconds\n", timeout_sec);
        }
      }
    }

    int pid = spawnDepthWorker(depthBackend);
    waitForDepthWorkerReady(pid, &segment->m_workerReadySem, timeout_sec);
  }


  // Flush SHM segment data to worker
  shm->flush(0, roundUp4K(lastOffset));

  uint32_t frameCount = 0;

  timingStatsAccumulator_t frontendFrameTime;
  timingStatsAccumulator_t backendFrameTime;

  while (true) {
    ++frameCount;
    if (endFrameCount && endFrameCount >= frameCount) {
      printf("Reached end condition.\n");
      break;
    }

    uint64_t startTime = currentTimeNs();

    // Signal worker to start depth processing
    sem_post(&segment->m_workAvailableSem);

    // Wait for processing to finish
    sem_wait(&segment->m_workFinishedSem);

    // Stats
    frontendFrameTime(deltaTimeMs(startTime, currentTimeNs()));
    backendFrameTime(segment->m_frameTimeMs);
    if (frameCount % 100 == 0) {
      printf("=== Frame %u\n", frameCount);
      printf("Frontend frame time: "); printTimingStats(frontendFrameTime);
      printf("Backend frame time: "); printTimingStats(backendFrameTime);
    }
  }
}

