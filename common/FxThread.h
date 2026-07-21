#pragma once
#include <functional> // std::invoke
#include <memory> // std::unique_ptr
#include <pthread.h>
#include <sched.h>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>

// A std::thread-like handle that pins each newly-created worker thread onto a configurable
// default CPU set at creation time. Because the affinity is baked into the pthread_attr_t
// before clone(), the child is born on the default set and never runs -- not even briefly --
// on cores outside it.
//
// The default set is captured once from sched_getaffinity() at static-initialization time,
// before main() has a chance to re-pin the calling thread. This composes with an isolcpus
// isolation strategy for free: the process's inherited affinity already excludes isolated
// core(s), so the captured default is exactly the "housekeeping" set that workers should
// inherit -- even after main() later pins itself to a reserved core. Override the default
// via setDefaultAffinity() if a different policy is wanted.
//
// This is the one thing std::thread cannot do: it constructs-and-starts atomically and
// never exposes a pthread_attr_t, so affinity can only be corrected after the thread is
// already running (via native_handle()), leaving a window on the wrong core.
//
// Only a subset of the std::thread interface is implemented: variadic callable+args
// construction, move-only, joinable()/join()/detach(). The destructor matches std::thread:
// std::terminate() if still joinable. When no default has been captured (sched_getaffinity
// failed, or an empty set), the child inherits normally.
class FxThread {
public:
  // Copy the process default CPU set (captured at static init) into `outSet`. Returns false
  // and zeroes `outSet` if no default is available, in which case affinity is left untouched.
  static bool getDefaultAffinity(cpu_set_t& outSet);

  // Override the default CPU set applied to threads spawned by the plain constructor.
  static void setDefaultAffinity(const cpu_set_t& cpuSet);

  FxThread() noexcept = default;

  // Spawn a worker born on the default CPU set (or inheriting, if none is available).
  template <typename Fn, typename... Args,
    typename = std::enable_if_t<!std::is_same<std::decay_t<Fn>, FxThread>::value>>
  explicit FxThread(Fn&& fn, Args&&... args) {
    cpu_set_t defaultSet;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    if (getDefaultAffinity(defaultSet))
      pthread_attr_setaffinity_np(&attr, sizeof defaultSet, &defaultSet);
    start(attr, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  // Escape hatch: spawn pinned to an explicit CPU set rather than the default -- for a thread
  // that has its own placement needs, or must deliberately be left on a specific core set.
  template <typename Fn, typename... Args>
  static FxThread withAffinity(const cpu_set_t& cpuSet, Fn&& fn, Args&&... args) {
    FxThread thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setaffinity_np(&attr, sizeof cpuSet, &cpuSet);
    thread.start(attr, std::forward<Fn>(fn), std::forward<Args>(args)...);
    return thread;
  }

  FxThread(FxThread&& other) noexcept :
    m_handle(other.m_handle),
    m_joinable(other.m_joinable) {
    other.m_joinable = false;
  }

  FxThread& operator=(FxThread&& other) noexcept {
    if (this != &other) {
      if (m_joinable)
        std::terminate(); // matches std::thread: overwriting a live handle is a bug
      m_handle = other.m_handle;
      m_joinable = other.m_joinable;
      other.m_joinable = false;
    }
    return *this;
  }

  FxThread(const FxThread&) = delete;
  FxThread& operator=(const FxThread&) = delete;

  ~FxThread() {
    if (m_joinable)
      std::terminate();
  }

  bool joinable() const noexcept { return m_joinable; }

  void join() {
    if (!m_joinable)
      throw std::system_error(std::make_error_code(std::errc::invalid_argument), "FxThread::join");
    pthread_join(m_handle, nullptr);
    m_joinable = false;
  }

  void detach() {
    if (!m_joinable)
      throw std::system_error(std::make_error_code(std::errc::invalid_argument), "FxThread::detach");
    pthread_detach(m_handle);
    m_joinable = false;
  }

  pthread_t nativeHandle() const noexcept { return m_handle; }

private:
  // Consumes `attr` (destroys it before returning). Decay-copies the callable and args into
  // a heap tuple exactly as std::thread does, then hands ownership to the new thread.
  template <typename Fn, typename... Args>
  void start(pthread_attr_t& attr, Fn&& fn, Args&&... args) {
    using State = std::tuple<std::decay_t<Fn>, std::decay_t<Args>...>;
    auto state = std::make_unique<State>(std::forward<Fn>(fn), std::forward<Args>(args)...);

    int rc = pthread_create(&m_handle, &attr, &FxThread::trampoline<State>, state.get());
    pthread_attr_destroy(&attr);
    if (rc != 0)
      throw std::system_error(rc, std::generic_category(), "FxThread: pthread_create");
    state.release(); // ownership transferred to the new thread's trampoline
    m_joinable = true;
  }

  template <typename State>
  static void* trampoline(void* raw) {
    std::unique_ptr<State> state(static_cast<State*>(raw));
    std::apply(
      [](auto&& fn, auto&&... a) {
        std::invoke(std::forward<decltype(fn)>(fn), std::forward<decltype(a)>(a)...);
      },
      *state);
    return nullptr;
  }

  pthread_t m_handle{};
  bool m_joinable = false;
};
