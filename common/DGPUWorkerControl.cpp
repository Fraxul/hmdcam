#include "common/DGPUWorkerControl.h"
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <string>


int spawnDGPUWorker() {

  char* exepath = realpath("/proc/self/exe", NULL);
  assert(exepath);

  std::string workerBin = std::string(dirname(exepath)) + "/dgpu-worker";
  free(exepath);

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

  printf("Spawning DGPU worker binary %s as PID %d\n", workerBin.c_str(), pid);
  return pid;
}

bool waitForDGPUWorkerReady(int pid, sem_t* sem, unsigned int timeout_sec) {

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  for (unsigned int i = 0; i < timeout_sec; ++i) {
    ts.tv_sec += 1;
    int res = sem_timedwait(sem, &ts);
    if (res == 0)
      return true;
    else if (errno != ETIMEDOUT) {
      perror("waitForDGPUWorkerReady(): sem_timedwait");
      return false;
    }
    int wstatus;
    if (waitpid(pid, &wstatus, WNOHANG) > 0) {
      if (WIFEXITED(wstatus)) {
        printf("DGPU worker exited; status %d\n", WEXITSTATUS(wstatus));
      } else if (WIFSIGNALED(wstatus)) {
        printf("DGPU worker exited; signal %d\n", WTERMSIG(wstatus));
      } else {
        printf("DGPU worker exited; unknown reason\n");
      }
      return false;
    }

  }
  printf("Timed out waiting for DGPU worker to initialize\n");
  return false;
}

bool spawnAndWaitForDGPUWorker(sem_t* sem, unsigned int timeout_sec) {
  if (getenv("DGPU_WORKER_ATTACH")) {
    // Debug support for launching the worker process externally (through a debugger or NSight)
    printf("Waiting for externally-spawned DGPU worker...\n");
    int res = sem_wait(sem);
    if (res != 0) {
      perror("sem_wait()");
      return false;
    }
    return true;
  }

  int pid = spawnDGPUWorker();
  return waitForDGPUWorkerReady(pid, sem, timeout_sec);
}

