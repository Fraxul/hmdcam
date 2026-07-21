#pragma once
#include <boost/function.hpp>
#include <boost/move/move.hpp>
#include <boost/thread/future.hpp> // packaged_task
#include <pthread.h>

// RAII utility for demoting the current thread to SCHED_IDLE.
class ScopedSchedIdle {
public:
  ScopedSchedIdle() :
    m_thread(pthread_self()) {

    pthread_getschedparam(m_thread, &m_prevPolicy, &m_prevParam);
    sched_param idle{};
    idle.sched_priority = 0;
    m_active = pthread_setschedparam(m_thread, SCHED_IDLE, &idle) == 0;
  }

  ~ScopedSchedIdle() {
    if (m_active)
      pthread_setschedparam(m_thread, m_prevPolicy, &m_prevParam);
  }

private:
  pthread_t m_thread;
  sched_param m_prevParam{};
  int m_prevPolicy = SCHED_OTHER;
  bool m_active = false;
};

// Promote the current thread to SCHED_FIFO and realtime priority.
// Uses SCHED_RESET_ON_FORK so that threads and processes launched from the promoted thread
// do not inherit the priority boost.
bool promoteCurrentThreadToRealtime(int rtPriority = 30);

namespace FxThreading {
namespace detail {
  void init();
  void shutdown();
} // namespace detail

// Runs the passed function like: for (size_t idx = start; idx < end; ++idx) { fn(idx); }
// Invocations of fn() are on different threads, up to the limit of hardware concurrency
// Blocking -- returns when all tasks have finished.
void runArrayTask(size_t start, size_t end, const boost::function<void(size_t)>& fn);

// Same as runArrayTask, but returns a function that must be called to ensure that the tasks have completed.
boost::function<void()> runArrayTaskAsync(size_t start, size_t end, const boost::function<void(size_t)>& fn);

// Runs the passed function async. Returned function can be called and will block until the task completes.
boost::function<void()> runTaskAsync(const boost::function<void()>& fn);

// Runs the passed function async. Does not block.
void runFunction(const boost::function<void()>& fn);

// Helper for running a boost::packaged_task
template <typename T>
void runPackagedTask(typename boost::packaged_task<T>& task) {
  boost::packaged_task<T>* pTask = new boost::packaged_task<T>(boost::move(task));

  runFunction([pTask]() { (*pTask)(); delete pTask; });
}

}; // namespace FxThreading
