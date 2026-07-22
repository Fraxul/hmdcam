#undef BOOST_NO_CXX11_RVALUE_REFERENCES

#include "FxThreading.h"
#include <boost/atomic.hpp>
#include <boost/bind/bind.hpp>
#include <boost/thread.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/shared_array.hpp>
#include <deque>
#include <sched.h>
#include <unistd.h>
#include <cstdio>
#include <cerrno>

using namespace boost;

namespace FxThreading {

static bool sInitialized = false;
thread_group sThreadGroup;
std::deque<function<void()>> sWorkQueue;
mutex sWorkQueueLock;
condition_variable sWorkAddedSignal;

void workerThread() {
  while (true) {
    unique_lock<mutex> lock(sWorkQueueLock);
    while (sWorkQueue.empty()) {
      sWorkAddedSignal.wait(lock);
    }

    function<void()> workFn = sWorkQueue.front();
    sWorkQueue.pop_front();

    lock.unlock(); // release while working

    workFn();
  }
}

void detail::init() {
  if (sInitialized)
    return;

  sInitialized = true;
  size_t threadCount = thread::hardware_concurrency();
  printf("FxThreading: Starting %zu threads to match hardware concurrency\n", threadCount);
  for (size_t i = 0; i < threadCount; ++i) {
    sThreadGroup.create_thread(&workerThread);
  }
}

void detail::shutdown() {
  if (!sInitialized)
    return;

  sInitialized = false;
  sThreadGroup.interrupt_all();
  sThreadGroup.join_all();
}

void arrayTaskWorkFn(const function<void(size_t)>& fn, shared_ptr<atomic<size_t>> counter, size_t endValue, shared_array<promise<bool>> donePromises, size_t promiseIdx) {
  while (true) {
    size_t idx = counter->fetch_add(1);
    if (idx >= endValue)
      break;

    fn(idx);
  }

  bool done = true;
  donePromises[promiseIdx].set_value(done);
}

void runArrayTask(size_t startValue, size_t endValue, const function<void(size_t)>& fn) {
  detail::init();

  shared_ptr<atomic<size_t>> counter(new atomic<size_t>(startValue));
  size_t count = sThreadGroup.size();

  shared_array<promise<bool>> promises(new promise<bool>[count]);
  scoped_array<unique_future<bool>> futures(new unique_future<bool>[count]);
  for (size_t i = 0; i < count; ++i) {
    futures[i] = promises[i].get_future();
  }

  {
    unique_lock<mutex> lock(sWorkQueueLock);
    for (size_t i = 0; i < count; ++i)
      sWorkQueue.push_back(boost::bind(&arrayTaskWorkFn, boost::ref(fn), counter, endValue, promises, i));
    sWorkAddedSignal.notify_all();
  }

  wait_for_all(futures.get(), futures.get() + count);
}

boost::function<void()> runArrayTaskAsync(size_t startValue, size_t endValue, const function<void(size_t)>& fn) {
  detail::init();

  shared_ptr<atomic<size_t>> counter(new atomic<size_t>(startValue));
  size_t count = sThreadGroup.size();

  shared_array<promise<bool>> promises(new promise<bool>[count]);
  shared_array<unique_future<bool>> futures(new unique_future<bool>[count]);
  for (size_t i = 0; i < count; ++i) {
    futures[i] = promises[i].get_future();
  }

  {
    unique_lock<mutex> lock(sWorkQueueLock);
    for (size_t i = 0; i < count; ++i)
      sWorkQueue.push_back(boost::bind(&arrayTaskWorkFn, fn, counter, endValue, promises, i));
    sWorkAddedSignal.notify_all();
  }

  return boost::function<void()>([futures, count]() { wait_for_all(futures.get(), futures.get() + count); });
}

boost::function<void()> runTaskAsync(const boost::function<void()>& fn) {
  detail::init();

  shared_ptr<promise<void>> done_promise(new promise<void>());
  shared_ptr<unique_future<void>> done_future(new unique_future<void>(done_promise->get_future()));

  {
    unique_lock<mutex> lock(sWorkQueueLock);
    sWorkQueue.push_back(boost::function<void()>([fn, done_promise]() { fn(); done_promise->set_value(); }));
    sWorkAddedSignal.notify_one();
  }

  return boost::function<void()>([done_future]() { done_future->get(); });
}

void runFunction(const boost::function<void()>& fn) {
  detail::init();

  {
    unique_lock<mutex> lock(sWorkQueueLock);
    sWorkQueue.push_back(fn);
    sWorkAddedSignal.notify_one();
  }
}

} // namespace FxThreading

// ----- Other utility functions -----

bool promoteCurrentThreadToRealtime(int rtPriority) {
  // Clamp the requested priority into the kernel's valid SCHED_FIFO range.
  const int prioMin = sched_get_priority_min(SCHED_FIFO);
  const int prioMax = sched_get_priority_max(SCHED_FIFO);
  if (rtPriority < prioMin) rtPriority = prioMin;
  if (rtPriority > prioMax) rtPriority = prioMax;

  // SCHED_RESET_ON_FORK: threads (and processes) this thread subsequently clones are reset to
  // SCHED_OTHER instead of inheriting SCHED_FIFO. The reset lives in the kernel's sched_fork(),
  // which is on the common copy_process() path for both fork() and pthread_create()'s
  // clone(CLONE_THREAD), so it covers raw std::thread/pthread_create spawns as well.
  struct sched_param param;
  memset(&param, 0, sizeof param);
  param.sched_priority = rtPriority;
  if (sched_setscheduler(0, SCHED_FIFO | SCHED_RESET_ON_FORK, &param) != 0) { // 0 == calling thread
    if (errno == EPERM)
      fprintf(stderr,
        "[perf] promoteCallingThreadRealtime: SCHED_FIFO denied (EPERM).\n"
        "       Ensure the current user has a nonzero `rtprio` rlimit.\n");
    else
      fprintf(stderr, "[perf] promoteCallingThreadRealtime: setscheduler failed: %s\n",
        strerror(errno));
    return false;
  }

  return true;
}

// Parse a Linux cpulist ("0-3,5,7") into a cpu_set_t. Returns the number of CPUs set.
static int parseCpuList(const char* text, cpu_set_t* out) {
  CPU_ZERO(out);
  int count = 0;
  for (const char* p = text; *p;) {
    while (*p == ',' || *p == ' ' || *p == '\n') ++p;
    if (!*p) break;
    char* end = nullptr;
    long lo = strtol(p, &end, 10);
    if (end == p) break; // malformed; stop
    p = end;
    long hi = lo;
    if (*p == '-') {
      hi = strtol(++p, &end, 10);
      if (end == p) break;
      p = end;
    }
    for (long c = lo; c <= hi && c < CPU_SETSIZE; ++c)
      if (c >= 0 && !CPU_ISSET((int) c, out)) {
        CPU_SET((int) c, out);
        ++count;
      }
  }
  return count;
}

// Read a cpulist sysfs file into a cpu_set_t. Returns count, or -1 if unreadable.
static int readCpuListFile(const char* path, cpu_set_t* out) {
  CPU_ZERO(out);
  FILE* f = fopen(path, "r");
  if (!f) return -1;
  char buf[512] = {0};
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = '\0';
  return parseCpuList(buf, out);
}

int pinCallingThreadToIsolatedCore() {
  cpu_set_t isolated, online;
  int nIsolated = readCpuListFile("/sys/devices/system/cpu/isolated", &isolated);
  readCpuListFile("/sys/devices/system/cpu/online", &online);

  if (nIsolated <= 0) {
    fprintf(stderr,
      "[perf] WARNING: no isolated CPU core (/sys/devices/system/cpu/isolated is empty).\n"
      "       Add to the kernel cmdline (highest core index N):\n"
      "         isolcpus=domain,managed_irq,N irqaffinity=0-(N-1)\n");
    // We skip recommending nohz_full=N rcu_nocbs=N because the Tegra kernel build doesn't support those flags.
    return -1;
  }

  // Pick the highest-numbered isolated core that is online
  int target = -1;
  for (int c = CPU_SETSIZE - 1; c >= 0; --c)
    if (CPU_ISSET(c, &isolated) && CPU_ISSET(c, &online)) {
      target = c;
      break;
    }
  if (target < 0) {
    fprintf(stderr, "[perf] WARNING: isolated core(s) present but none online; not pinning.\n");
    return -1;
  }

#if 0
  // nohz_full isn't supported on the Tegra kernel build as of L4T r36.4 (5.15.148-tegra)
  {
    cpu_set_t nohz;
    int nNohz = readCpuListFile("/sys/devices/system/cpu/nohz_full", &nohz);
    if (nNohz <= 0 || !CPU_ISSET(target, &nohz))
      fprintf(stderr, "[perf] NOTE: CPU %d is isolated but not nohz_full; add nohz_full=%d "
                      "to also drop the scheduler-tick jitter.\n",
        target, target);
  }
#endif

  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(target, &one);
  if (sched_setaffinity(0, sizeof(one), &one) != 0) { // 0 == calling thread
    fprintf(stderr, "[perf] ERROR: pin to CPU %d failed: %s\n", target, strerror(errno));
    return -1;
  }
  // Verify — a cpuset or an offline race can silently clamp the request.
  cpu_set_t got;
  CPU_ZERO(&got);
  if (sched_getaffinity(0, sizeof(got), &got) == 0 &&
    (CPU_COUNT(&got) != 1 || !CPU_ISSET(target, &got))) {
    fprintf(stderr, "[perf] WARNING: affinity to CPU %d didn't stick (cpuset override?).\n", target);
    return -1;
  }

  // fprintf(stderr, "[perf] Thread %u pinned to isolated CPU %d.\n", currentTid(), target);
  return target;
}
