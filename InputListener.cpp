#include "InputListener.h"
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>

#include <linux/input.h>
#include <sys/epoll.h>
#include <libudev.h>

#include <map>
#include <set>
#include <string>
#include <atomic>

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

std::atomic<bool> buttonState[kButtonCount];
static std::map<std::string, int> keyboardDevices;
static int epoll_fd = -1;

bool testButton(EButton button) {
  assert(button < kButtonCount);
  return buttonState[button].exchange(false);
}

void udevProcessDevice(struct udev_device* dev) {
  if (!dev)
    return;

  {
    const char* devicePath = udev_device_get_devnode(dev);
    if (!devicePath) {
      goto cleanup;
    }

    if (!udev_device_get_property_value(dev, "ID_INPUT_KEYBOARD")) {
      goto cleanup;
    }

    const char* action = udev_device_get_action(dev);
    if (!action)
      action = "exists";

    printf("Input device %s: %s\n", devicePath, action);

    if ((!strcmp(action, "add")) || (!strcmp(action, "exists"))) {
      int& fd = keyboardDevices[devicePath];

      fd = open(devicePath, O_RDONLY | O_CLOEXEC);
      if (fd < 0) {
        printf("Unable to open event device %s. Device will be ignored.\n", devicePath);
        goto cleanup;
      }

      // Grab input
      if(ioctl(fd, EVIOCGRAB, (void*)1) == -1) {
        printf("Failed to grab input for device %s: %s\n", devicePath, strerror(errno));
        close(fd);
        fd = -1;
        goto cleanup;
      }

      // Device is open and grabbed, add it to the epoll set
      struct epoll_event ev;
      ev.events = EPOLLIN;
      ev.data.fd = fd;
      if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) < 0) {
        printf("epoll_ctl() failed adding device fd: %s\n", strerror(errno));
        close(fd);
        fd = -1;
        goto cleanup;
      }

    } else if (!strcmp(action, "remove")) {
      int fd = keyboardDevices[devicePath];
      printf("Device removed: %s\n", devicePath);

      if (fd >= 0) {
        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
        close(fd);
      }
      keyboardDevices.erase(devicePath);
    } else {
      printf("Unhandled event type %s\n", action);
    }
  }

cleanup:
  udev_device_unref(dev);
}


void* inputListenerThread(void*) {

  // Init-once for static button state
  for (size_t buttonIdx = 0; buttonIdx < kButtonCount; ++buttonIdx) {
    std::atomic_init<bool>(&buttonState[buttonIdx], false);
  }

  // Set up epoll
  epoll_fd = epoll_create1(0);
  if (epoll_fd < 0) {
    die("epoll_create1() failed: %s", strerror(errno));
  }

  // Set up udev monitoring
  struct udev* udev = udev_new();
  struct udev_monitor* mon = udev_monitor_new_from_netlink(udev, "udev");
  
  udev_monitor_filter_add_match_subsystem_devtype(mon, "input", NULL);
  udev_monitor_enable_receiving(mon);

  int udev_fd = udev_monitor_get_fd(mon);

  {
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = udev_fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, udev_fd, &ev) < 0) {
      die("epoll_ctl() failed to add inotify fd: %s", strerror(errno));
    }
  }

  // Initial device scan
  struct udev_enumerate* enumerate = udev_enumerate_new(udev);

  udev_enumerate_add_match_subsystem(enumerate, "input");
  udev_enumerate_add_match_property(enumerate, "ID_INPUT_KEYBOARD", "1");

  udev_enumerate_scan_devices(enumerate);

  struct udev_list_entry* devices = udev_enumerate_get_list_entry(enumerate);
  struct udev_list_entry* entry;

  udev_list_entry_foreach(entry, devices) {
      const char* path = udev_list_entry_get_name(entry);
      struct udev_device* dev = udev_device_new_from_syspath(udev, path);
      udevProcessDevice(dev);
  }

  udev_enumerate_unref(enumerate);

  // Process events
  while (true) {

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

      if (fd == udev_fd) {
        struct udev_device* dev = udev_monitor_receive_device(mon);
        udevProcessDevice(dev);

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

        // Previously using keycodes for a presentation remote, where:
        // Up    => "Blank Screen" (b)
        // Down  => "Start/Stop Presentation" (alternates between Shift+F5 and ESC)
        // Left  => "Previous Slide" (Page Up)
        // Right => "Next Slide" (Page Down)

        // Now using keycodes for a media remote. More direct mapping.
        switch (ev.code) {
          case KEY_DOWN:
          //case KEY_ESC:
          //case KEY_F5:
            buttonState[kButtonDown].store(true);
            break;

          case KEY_LEFT:
          //case KEY_PAGEUP:
            buttonState[kButtonLeft].store(true);
            break;

          case KEY_RIGHT:
          //case KEY_PAGEDOWN:
            buttonState[kButtonRight].store(true);
            break;

          case KEY_UP:
          //case KEY_B:
            buttonState[kButtonUp].store(true);
            break;

          case KEY_POWER:
            buttonState[kButtonPower].store(true);
            break;

          case KEY_HOMEPAGE:
            buttonState[kButtonHome].store(true);
            break;

          case KEY_COMPOSE:
            buttonState[kButtonMenu].store(true);
            break;

          case KEY_ESC:
          case KEY_BACK:
            buttonState[kButtonBack].store(true);
            break;

          case KEY_ENTER:
            buttonState[kButtonOK].store(true);
            break;

          default:
            break;
        };
      } // keyboard event processing

    } // epoll_wait fd loop

  } // epoll event loop

  close(udev_fd);
  close(epoll_fd);

  udev_monitor_unref(mon);
  udev_unref(udev);

  return NULL;
}

void startInputListenerThread() {
  pthread_t thread;
  pthread_create(&thread, NULL, inputListenerThread, NULL);
}

