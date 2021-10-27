#include "common/DepthWorkerControl.h"
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <string>
#include <limits.h>

static DepthWorkerBackend s_currentBackend = kDepthWorkerNone;

DepthWorkerBackend currentDepthWorkerBackend() {
  return s_currentBackend;
}

int spawnDepthWorker(DepthWorkerBackend backend) {

  char* exepath = realpath("/proc/self/exe", NULL);
  assert(exepath);

  std::string workerBin = std::string(dirname(exepath));
  free(exepath);

  switch (backend) {
    case kDepthWorkerNone:
      assert(false && "spawnDepthWorker: can't spawn process for backend type kDepthWorkerNone");
      break;

    case kDepthWorkerDGPU:
      workerBin += "/dgpu-worker";
      break;

    case kDepthWorkerDepthAI:
      workerBin += "/depthai-worker";
      break;

    default:
      assert(false && "spawnDepthWorker: invalid backend enum");
  };
  s_currentBackend = backend;

  int pid = fork();
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

bool waitForDepthWorkerReady(int pid, sem_t* sem, unsigned int timeout_sec) {

  {
    const char* timeoutEnvStr = getenv("DEPTH_WORKER_TIMEOUT");
    if (timeoutEnvStr) {
      int timeoutEnv = atoi(timeoutEnvStr);
      if (timeoutEnv <= 0) {
        timeout_sec = UINT_MAX;
        fprintf(stderr, "waitForDepthWorkerReady: DEPTH_WORKER_TIMEOUT <= 0, waiting forever\n");
      } else {
        timeout_sec = timeoutEnv;
        fprintf(stderr, "waitForDepthWorkerReady: DEPTH_WORKER_TIMEOUT is %u seconds\n", timeout_sec);
      }
    }
  }

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  for (unsigned int i = 0; i < timeout_sec; ++i) {
    ts.tv_sec += 1;
    int res = sem_timedwait(sem, &ts);
    if (res == 0)
      return true;
    else if (errno != ETIMEDOUT) {
      perror("waitForDepthWorkerReady(): sem_timedwait");
      return false;
    }
    int wstatus;
    if (waitpid(pid, &wstatus, WNOHANG) > 0) {
      if (WIFEXITED(wstatus)) {
        printf("Depth worker exited; status %d\n", WEXITSTATUS(wstatus));
      } else if (WIFSIGNALED(wstatus)) {
        printf("Depth worker exited; signal %d\n", WTERMSIG(wstatus));
      } else {
        printf("Depth worker exited; unknown reason\n");
      }
      return false;
    }

  }
  printf("Timed out waiting for Depth worker to initialize\n");
  return false;
}

bool spawnAndWaitForDepthWorker(DepthWorkerBackend backend, sem_t* sem, unsigned int timeout_sec) {
  s_currentBackend = backend;
  if (getenv("DEPTH_WORKER_ATTACH")) {
    // Debug support for launching the worker process externally (through a debugger or NSight)
    printf("Waiting for externally-spawned depth worker...\n");
    int res = sem_wait(sem);
    if (res != 0) {
      perror("sem_wait()");
      return false;
    }
    return true;
  }

  int pid = spawnDepthWorker(backend);
  return waitForDepthWorkerReady(pid, sem, timeout_sec);
}

DepthWorkerBackend depthBackendStringToEnum(const char* backendStr) {
  if (!strcasecmp(backendStr, "none")) {
    return kDepthWorkerNone;
  } else if ((!strcasecmp(backendStr, "dgpu")) || (!strcasecmp(backendStr, "cuda"))) {
    return kDepthWorkerDGPU;
  } else if ((!strcasecmp(backendStr, "depthai")) || (!strcasecmp(backendStr, "depth-ai"))) {
    return kDepthWorkerDepthAI;
  } else {
    fprintf(stderr, "depthBackendStringToEnum: unrecognized worker type \"%s\"\n", backendStr);
    return kDepthWorkerNone;
  }
}

