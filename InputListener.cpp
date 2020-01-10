#include "InputListener.h"
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <glob.h>
#include <assert.h>
#include <pthread.h>

#include <linux/input.h>
#include <sys/epoll.h>
#include <sys/inotify.h>

#include <map>
#include <set>
#include <string>
#include <atomic>

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

std::atomic<bool> buttonState[kButtonCount];

bool testButton(EButton button) {
  assert(button < kButtonCount);
  return buttonState[button].exchange(false);
}

void* inputListenerThread(void*) {

  // Init-once for static button state
  for (size_t buttonIdx = 0; buttonIdx < kButtonCount; ++buttonIdx) {
    std::atomic_init<bool>(&buttonState[buttonIdx], false);
  }

  int inotify_fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  if (inotify_fd < 0) {
    die("inotify_init1() failed: %s", strerror(errno));
  }

  int inotify_watch_descriptor = inotify_add_watch(inotify_fd, "/dev/input/by-id", IN_CREATE | IN_DELETE);
  if (inotify_watch_descriptor < 0) {
    die("inotify_add_watch() failed: %s", strerror(errno));
  }

  int epoll_fd = epoll_create1(0);
  if (epoll_fd < 0) {
    die("epoll_create1() failed: %s", strerror(errno));
  }

  {
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = inotify_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, inotify_fd, &ev) < 0) {
      die("epoll_ctl() failed to add inotify fd: %s", strerror(errno));
    }
  }

  std::map<std::string, int> keyboardDevices;

  bool shouldRescanDevices = true;

  while (true) {

    if (shouldRescanDevices) {
      shouldRescanDevices = false;

      std::set<std::string> scanResults;
      {
        glob_t globResult;
        memset(&globResult, 0, sizeof(glob_t));
        int globRes = glob("/dev/input/by-id/*-kbd", GLOB_NOSORT, NULL, &globResult);

        if (globRes == GLOB_NOMATCH) {
          // no matches, but that's OK
        } else if (globRes == 0) {
          for (size_t matchIdx = 0; matchIdx < globResult.gl_pathc; ++matchIdx) {
            scanResults.insert(globResult.gl_pathv[matchIdx]);
          }
        } else {
          if (errno == GLOB_NOMATCH) {}
          else {
            die("glob(): %s", strerror(errno));
          }
        }
        globfree(&globResult); 
      }

      for (const std::string& devicePath : scanResults) {
        auto deviceMapIt = keyboardDevices.find(devicePath);
        if (deviceMapIt != keyboardDevices.end()) {
          // Device is either already open or a previous open attempt failed (in which case we ignore it until it's unplugged)
          continue;
        }

        // Device appears to be new
        printf("New device path: %s\n", devicePath.c_str());
        int& fd = keyboardDevices[devicePath];

        fd = open(devicePath.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
          printf("Unable to open event device %s. Device will be ignored.\n", devicePath.c_str());
          continue;
        }

        // Grab input
        if(ioctl(fd, EVIOCGRAB, (void*)1) == -1) {
          printf("Failed to grab input for device %s: %s\n", devicePath.c_str(), strerror(errno));
          close(fd);
          fd = -1;
          continue;
        }

        // Device is open and grabbed, add it to the epoll set
        struct epoll_event ev;
        ev.events = EPOLLIN;
        ev.data.fd = fd;
        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) < 0) {
          printf("epoll_ctl() failed adding device fd: %s\n", strerror(errno));
          close(fd);
          fd = -1;
          continue;
        }
      }

      // Check for devices that have been removed
      for (auto deviceMapIt = keyboardDevices.begin(); deviceMapIt != keyboardDevices.end(); ) {
        const std::string& devicePath = deviceMapIt->first;
        int fd = deviceMapIt->second;
        if (scanResults.find(devicePath) != scanResults.end()) {
          ++deviceMapIt;
          continue; // Device still exists
        }

        printf("Device removed: %s\n", devicePath.c_str());

        if (fd >= 0) {
          epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
          close(fd);
        }

        keyboardDevices.erase(deviceMapIt);
        deviceMapIt = keyboardDevices.begin(); // iterator invalidated
      }

    } /// device rescan finished


    const size_t MAX_EVENTS = 32;
    struct epoll_event events[MAX_EVENTS];
    int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
    if (nfds == -1) {
      if (errno == EINTR)
        continue; // ok

      die("epoll_wait() failed: %s", strerror(errno));
    }

    for (ssize_t evIdx = 0; evIdx < nfds; ++evIdx) {
      int fd = events[evIdx].data.fd;

      if (fd == inotify_fd) {
        shouldRescanDevices = true;
        // drain the inotify buffer. we don't actually care about the contents since we just use it to trigger a scan
        char buf[4096];
        while (true) {
          ssize_t readlen = read(inotify_fd, buf, 4096);
          //printf("inotify fd read returned %zd\n", readlen);
          if (readlen < 0) {
            if (errno == EAGAIN) {
              break;
            } else {
              printf("read() on inotify FD: %s\n", strerror(errno));
            }
          }
        }
      } else {

        struct input_event ev;
        memset(&ev, 0, sizeof(ev));

        if (read(fd, &ev, sizeof(ev)) == -1) {
          if (errno == EAGAIN)
            continue;

          int readErrno = errno;
          if (readErrno != ENODEV) { // ENODEV is expected when a device is removed
            printf("read() failed: %s\n", strerror(readErrno));
          }

          // Close the device on read failure.
          epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
          close(fd);

          // Find the entry in the device map and remove the FD association
          for (auto it = keyboardDevices.begin(); it != keyboardDevices.end(); ++it) {
            if (it->second == fd) {
              printf("Device %s closed due to %s\n", it->first.c_str(), readErrno == ENODEV ? "removal" : "read error");
              it->second = -1;
              break;
            }
          }

          continue;
        }
        
        if (ev.type != EV_KEY) continue; // Looking for KEY events...
        if (ev.value != 1) continue; // ...where the key is being pressed. (value={0, 1, 2} is {released, pressed, repeat})

        // Keycodes for a presentation remote, where:
        // Up    => "Blank Screen" (b)
        // Down  => "Start/Stop Presentation" (alternates between Shift+F5 and ESC)
        // Left  => "Previous Slide" (Page Up)
        // Right => "Next Slide" (Page Down)
        switch (ev.code) {
          case KEY_DOWN:
          case KEY_ESC:
          case KEY_F5:
            //printf("Down\n");
            buttonState[kButtonDown].store(true);
            break;

          case KEY_LEFT:
          case KEY_PAGEUP: //
            //printf("Left\n");
            buttonState[kButtonLeft].store(true);
            break;

          case KEY_RIGHT:
          case KEY_PAGEDOWN:
            //printf("Right\n");
            buttonState[kButtonRight].store(true);
            break;

          case KEY_UP:
          case KEY_B:
            //printf("Up\n");
            buttonState[kButtonUp].store(true);
            break;

          default:
            break;
        };
      } // keyboard event processing

    } // epoll_wait fd loop

  } // epoll event loop

  close(inotify_fd);
  close(epoll_fd);

  return NULL;
}

void startInputListenerThread() {
  pthread_t thread;
  pthread_create(&thread, NULL, inputListenerThread, NULL);
}

