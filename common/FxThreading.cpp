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
