#pragma once
#include <semaphore.h>

enum DepthWorkerBackend {
  kDepthWorkerNone,
  kDepthWorkerDGPU,
  kDepthWorkerDepthAI
};

// returns PID
int spawnDepthWorker(DepthWorkerBackend backend);
bool waitForDepthWorkerReady(int pid, sem_t* sem, unsigned int timeout_sec);
bool spawnAndWaitForDepthWorker(DepthWorkerBackend backend, sem_t* sem, unsigned int timeout_sec = 5);
DepthWorkerBackend depthBackendStringToEnum(const char* backendStr);
DepthWorkerBackend currentDepthWorkerBackend(); // updated by spawnDepthWorker

