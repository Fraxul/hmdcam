#pragma once
#include <semaphore.h>

// returns PID
int spawnDGPUWorker();
bool waitForDGPUWorkerReady(int pid, sem_t* sem, unsigned int timeout_sec);
bool spawnAndWaitForDGPUWorker(sem_t* sem, unsigned int timeout_sec = 5);

