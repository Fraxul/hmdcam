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
static bool debugLogKeyEvents = false; // LOG_KEY_EVENTS environment variable

static const char* inputPropertyName(unsigned int v);
static const char* inputKeyName(unsigned int v);

bool testButton(EButton button) {
  assert(button < kButtonCount);
  return buttonState[button].exchange(false);
}

void clearButtons() {
  for (size_t i = 0; i < kButtonCount; ++i) {
    buttonState[i].exchange(false);
  }
}

void udevProcessDevice(struct udev_device* dev) {
  if (!dev)
    return;

  {
    const char* devicePath = udev_device_get_devnode(dev);
    if (!devicePath) {
      goto cleanup;
    }

    {
      const char* deviceType = udev_device_get_property_value(dev, "ID_TYPE");
      if ((!deviceType) || strcmp(deviceType, "hid") != 0) {
        goto cleanup;
      }
    }

    if (!udev_device_get_property_value(dev, "ID_INPUT_KEY")) {
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
  pthread_setname_np(pthread_self(), "InputListener");

  {
    char* ev = getenv("LOG_KEY_EVENTS");
    debugLogKeyEvents = (ev && atoi(ev) != 0);
  }

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
  udev_enumerate_add_match_property(enumerate, "ID_TYPE", "hid");
  udev_enumerate_add_match_property(enumerate, "ID_INPUT_KEY", "1");

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

        if (debugLogKeyEvents) {
          printf("InputListener: Key pressed: hex=%3x dec=%4u %s\n", ev.code, ev.code, inputKeyName(ev.code));
        }

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

// Ignore non-use of the enum-to-string functions below here
#pragma clang diagnostic ignored "-Wunused-function"

static const char* inputPropertyName(unsigned int v) {
  static char buf[64];
  switch (v) {
    case INPUT_PROP_POINTER: return "INPUT_PROP_POINTER";
    case INPUT_PROP_DIRECT: return "INPUT_PROP_DIRECT";
    case INPUT_PROP_BUTTONPAD: return "INPUT_PROP_BUTTONPAD";
    case INPUT_PROP_SEMI_MT: return "INPUT_PROP_SEMI_MT";
    case INPUT_PROP_TOPBUTTONPAD: return "INPUT_PROP_TOPBUTTONPAD";
    case INPUT_PROP_POINTING_STICK: return "INPUT_PROP_POINTING_STICK";
    case INPUT_PROP_ACCELEROMETER: return "INPUT_PROP_ACCELEROMETER";
    case INPUT_PROP_MAX: return "INPUT_PROP_MAX";
    default:
      break;
  }
  snprintf(buf, 64, "unknown 0x%x / %d", v, v);
  buf[63] = '\0';
  return buf;
}

static const char* inputKeyName(unsigned int v) {
  // grep -oE "#define\s+([a-zA-Z0-9_]+)\s+([0-9]+|0[xX][0-9a-fA-F]+)" /usr/include/linux/input-event-codes.h | awk '{print "case " $2 ": return \""$2"\";"}'
  static char buf[64];
  switch (v) {
/* Synchronization events
    case SYN_REPORT: return "SYN_REPORT";
    case SYN_CONFIG: return "SYN_CONFIG";
    case SYN_MT_REPORT: return "SYN_MT_REPORT";
    case SYN_DROPPED: return "SYN_DROPPED";
    case SYN_MAX: return "SYN_MAX";
*/
    case KEY_RESERVED: return "KEY_RESERVED";
    case KEY_ESC: return "KEY_ESC";
    case KEY_1: return "KEY_1";
    case KEY_2: return "KEY_2";
    case KEY_3: return "KEY_3";
    case KEY_4: return "KEY_4";
    case KEY_5: return "KEY_5";
    case KEY_6: return "KEY_6";
    case KEY_7: return "KEY_7";
    case KEY_8: return "KEY_8";
    case KEY_9: return "KEY_9";
    case KEY_0: return "KEY_0";
    case KEY_MINUS: return "KEY_MINUS";
    case KEY_EQUAL: return "KEY_EQUAL";
    case KEY_BACKSPACE: return "KEY_BACKSPACE";
    case KEY_TAB: return "KEY_TAB";
    case KEY_Q: return "KEY_Q";
    case KEY_W: return "KEY_W";
    case KEY_E: return "KEY_E";
    case KEY_R: return "KEY_R";
    case KEY_T: return "KEY_T";
    case KEY_Y: return "KEY_Y";
    case KEY_U: return "KEY_U";
    case KEY_I: return "KEY_I";
    case KEY_O: return "KEY_O";
    case KEY_P: return "KEY_P";
    case KEY_LEFTBRACE: return "KEY_LEFTBRACE";
    case KEY_RIGHTBRACE: return "KEY_RIGHTBRACE";
    case KEY_ENTER: return "KEY_ENTER";
    case KEY_LEFTCTRL: return "KEY_LEFTCTRL";
    case KEY_A: return "KEY_A";
    case KEY_S: return "KEY_S";
    case KEY_D: return "KEY_D";
    case KEY_F: return "KEY_F";
    case KEY_G: return "KEY_G";
    case KEY_H: return "KEY_H";
    case KEY_J: return "KEY_J";
    case KEY_K: return "KEY_K";
    case KEY_L: return "KEY_L";
    case KEY_SEMICOLON: return "KEY_SEMICOLON";
    case KEY_APOSTROPHE: return "KEY_APOSTROPHE";
    case KEY_GRAVE: return "KEY_GRAVE";
    case KEY_LEFTSHIFT: return "KEY_LEFTSHIFT";
    case KEY_BACKSLASH: return "KEY_BACKSLASH";
    case KEY_Z: return "KEY_Z";
    case KEY_X: return "KEY_X";
    case KEY_C: return "KEY_C";
    case KEY_V: return "KEY_V";
    case KEY_B: return "KEY_B";
    case KEY_N: return "KEY_N";
    case KEY_M: return "KEY_M";
    case KEY_COMMA: return "KEY_COMMA";
    case KEY_DOT: return "KEY_DOT";
    case KEY_SLASH: return "KEY_SLASH";
    case KEY_RIGHTSHIFT: return "KEY_RIGHTSHIFT";
    case KEY_KPASTERISK: return "KEY_KPASTERISK";
    case KEY_LEFTALT: return "KEY_LEFTALT";
    case KEY_SPACE: return "KEY_SPACE";
    case KEY_CAPSLOCK: return "KEY_CAPSLOCK";
    case KEY_F1: return "KEY_F1";
    case KEY_F2: return "KEY_F2";
    case KEY_F3: return "KEY_F3";
    case KEY_F4: return "KEY_F4";
    case KEY_F5: return "KEY_F5";
    case KEY_F6: return "KEY_F6";
    case KEY_F7: return "KEY_F7";
    case KEY_F8: return "KEY_F8";
    case KEY_F9: return "KEY_F9";
    case KEY_F10: return "KEY_F10";
    case KEY_NUMLOCK: return "KEY_NUMLOCK";
    case KEY_SCROLLLOCK: return "KEY_SCROLLLOCK";
    case KEY_KP7: return "KEY_KP7";
    case KEY_KP8: return "KEY_KP8";
    case KEY_KP9: return "KEY_KP9";
    case KEY_KPMINUS: return "KEY_KPMINUS";
    case KEY_KP4: return "KEY_KP4";
    case KEY_KP5: return "KEY_KP5";
    case KEY_KP6: return "KEY_KP6";
    case KEY_KPPLUS: return "KEY_KPPLUS";
    case KEY_KP1: return "KEY_KP1";
    case KEY_KP2: return "KEY_KP2";
    case KEY_KP3: return "KEY_KP3";
    case KEY_KP0: return "KEY_KP0";
    case KEY_KPDOT: return "KEY_KPDOT";
    case KEY_ZENKAKUHANKAKU: return "KEY_ZENKAKUHANKAKU";
    case KEY_102ND: return "KEY_102ND";
    case KEY_F11: return "KEY_F11";
    case KEY_F12: return "KEY_F12";
    case KEY_RO: return "KEY_RO";
    case KEY_KATAKANA: return "KEY_KATAKANA";
    case KEY_HIRAGANA: return "KEY_HIRAGANA";
    case KEY_HENKAN: return "KEY_HENKAN";
    case KEY_KATAKANAHIRAGANA: return "KEY_KATAKANAHIRAGANA";
    case KEY_MUHENKAN: return "KEY_MUHENKAN";
    case KEY_KPJPCOMMA: return "KEY_KPJPCOMMA";
    case KEY_KPENTER: return "KEY_KPENTER";
    case KEY_RIGHTCTRL: return "KEY_RIGHTCTRL";
    case KEY_KPSLASH: return "KEY_KPSLASH";
    case KEY_SYSRQ: return "KEY_SYSRQ";
    case KEY_RIGHTALT: return "KEY_RIGHTALT";
    case KEY_LINEFEED: return "KEY_LINEFEED";
    case KEY_HOME: return "KEY_HOME";
    case KEY_UP: return "KEY_UP";
    case KEY_PAGEUP: return "KEY_PAGEUP";
    case KEY_LEFT: return "KEY_LEFT";
    case KEY_RIGHT: return "KEY_RIGHT";
    case KEY_END: return "KEY_END";
    case KEY_DOWN: return "KEY_DOWN";
    case KEY_PAGEDOWN: return "KEY_PAGEDOWN";
    case KEY_INSERT: return "KEY_INSERT";
    case KEY_DELETE: return "KEY_DELETE";
    case KEY_MACRO: return "KEY_MACRO";
    case KEY_MUTE: return "KEY_MUTE";
    case KEY_VOLUMEDOWN: return "KEY_VOLUMEDOWN";
    case KEY_VOLUMEUP: return "KEY_VOLUMEUP";
    case KEY_POWER: return "KEY_POWER";
    case KEY_KPEQUAL: return "KEY_KPEQUAL";
    case KEY_KPPLUSMINUS: return "KEY_KPPLUSMINUS";
    case KEY_PAUSE: return "KEY_PAUSE";
    case KEY_SCALE: return "KEY_SCALE";
    case KEY_KPCOMMA: return "KEY_KPCOMMA";
    case KEY_HANGEUL: return "KEY_HANGEUL";
    case KEY_HANJA: return "KEY_HANJA";
    case KEY_YEN: return "KEY_YEN";
    case KEY_LEFTMETA: return "KEY_LEFTMETA";
    case KEY_RIGHTMETA: return "KEY_RIGHTMETA";
    case KEY_COMPOSE: return "KEY_COMPOSE";
    case KEY_STOP: return "KEY_STOP";
    case KEY_AGAIN: return "KEY_AGAIN";
    case KEY_PROPS: return "KEY_PROPS";
    case KEY_UNDO: return "KEY_UNDO";
    case KEY_FRONT: return "KEY_FRONT";
    case KEY_COPY: return "KEY_COPY";
    case KEY_OPEN: return "KEY_OPEN";
    case KEY_PASTE: return "KEY_PASTE";
    case KEY_FIND: return "KEY_FIND";
    case KEY_CUT: return "KEY_CUT";
    case KEY_HELP: return "KEY_HELP";
    case KEY_MENU: return "KEY_MENU";
    case KEY_CALC: return "KEY_CALC";
    case KEY_SETUP: return "KEY_SETUP";
    case KEY_SLEEP: return "KEY_SLEEP";
    case KEY_WAKEUP: return "KEY_WAKEUP";
    case KEY_FILE: return "KEY_FILE";
    case KEY_SENDFILE: return "KEY_SENDFILE";
    case KEY_DELETEFILE: return "KEY_DELETEFILE";
    case KEY_XFER: return "KEY_XFER";
    case KEY_PROG1: return "KEY_PROG1";
    case KEY_PROG2: return "KEY_PROG2";
    case KEY_WWW: return "KEY_WWW";
    case KEY_MSDOS: return "KEY_MSDOS";
    case KEY_COFFEE: return "KEY_COFFEE";
    case KEY_ROTATE_DISPLAY: return "KEY_ROTATE_DISPLAY";
    case KEY_CYCLEWINDOWS: return "KEY_CYCLEWINDOWS";
    case KEY_MAIL: return "KEY_MAIL";
    case KEY_BOOKMARKS: return "KEY_BOOKMARKS";
    case KEY_COMPUTER: return "KEY_COMPUTER";
    case KEY_BACK: return "KEY_BACK";
    case KEY_FORWARD: return "KEY_FORWARD";
    case KEY_CLOSECD: return "KEY_CLOSECD";
    case KEY_EJECTCD: return "KEY_EJECTCD";
    case KEY_EJECTCLOSECD: return "KEY_EJECTCLOSECD";
    case KEY_NEXTSONG: return "KEY_NEXTSONG";
    case KEY_PLAYPAUSE: return "KEY_PLAYPAUSE";
    case KEY_PREVIOUSSONG: return "KEY_PREVIOUSSONG";
    case KEY_STOPCD: return "KEY_STOPCD";
    case KEY_RECORD: return "KEY_RECORD";
    case KEY_REWIND: return "KEY_REWIND";
    case KEY_PHONE: return "KEY_PHONE";
    case KEY_ISO: return "KEY_ISO";
    case KEY_CONFIG: return "KEY_CONFIG";
    case KEY_HOMEPAGE: return "KEY_HOMEPAGE";
    case KEY_REFRESH: return "KEY_REFRESH";
    case KEY_EXIT: return "KEY_EXIT";
    case KEY_MOVE: return "KEY_MOVE";
    case KEY_EDIT: return "KEY_EDIT";
    case KEY_SCROLLUP: return "KEY_SCROLLUP";
    case KEY_SCROLLDOWN: return "KEY_SCROLLDOWN";
    case KEY_KPLEFTPAREN: return "KEY_KPLEFTPAREN";
    case KEY_KPRIGHTPAREN: return "KEY_KPRIGHTPAREN";
    case KEY_NEW: return "KEY_NEW";
    case KEY_REDO: return "KEY_REDO";
    case KEY_F13: return "KEY_F13";
    case KEY_F14: return "KEY_F14";
    case KEY_F15: return "KEY_F15";
    case KEY_F16: return "KEY_F16";
    case KEY_F17: return "KEY_F17";
    case KEY_F18: return "KEY_F18";
    case KEY_F19: return "KEY_F19";
    case KEY_F20: return "KEY_F20";
    case KEY_F21: return "KEY_F21";
    case KEY_F22: return "KEY_F22";
    case KEY_F23: return "KEY_F23";
    case KEY_F24: return "KEY_F24";
    case KEY_PLAYCD: return "KEY_PLAYCD";
    case KEY_PAUSECD: return "KEY_PAUSECD";
    case KEY_PROG3: return "KEY_PROG3";
    case KEY_PROG4: return "KEY_PROG4";
    case KEY_ALL_APPLICATIONS: return "KEY_ALL_APPLICATIONS";
    case KEY_SUSPEND: return "KEY_SUSPEND";
    case KEY_CLOSE: return "KEY_CLOSE";
    case KEY_PLAY: return "KEY_PLAY";
    case KEY_FASTFORWARD: return "KEY_FASTFORWARD";
    case KEY_BASSBOOST: return "KEY_BASSBOOST";
    case KEY_PRINT: return "KEY_PRINT";
    case KEY_HP: return "KEY_HP";
    case KEY_CAMERA: return "KEY_CAMERA";
    case KEY_SOUND: return "KEY_SOUND";
    case KEY_QUESTION: return "KEY_QUESTION";
    case KEY_EMAIL: return "KEY_EMAIL";
    case KEY_CHAT: return "KEY_CHAT";
    case KEY_SEARCH: return "KEY_SEARCH";
    case KEY_CONNECT: return "KEY_CONNECT";
    case KEY_FINANCE: return "KEY_FINANCE";
    case KEY_SPORT: return "KEY_SPORT";
    case KEY_SHOP: return "KEY_SHOP";
    case KEY_ALTERASE: return "KEY_ALTERASE";
    case KEY_CANCEL: return "KEY_CANCEL";
    case KEY_BRIGHTNESSDOWN: return "KEY_BRIGHTNESSDOWN";
    case KEY_BRIGHTNESSUP: return "KEY_BRIGHTNESSUP";
    case KEY_MEDIA: return "KEY_MEDIA";
    case KEY_SWITCHVIDEOMODE: return "KEY_SWITCHVIDEOMODE";
    case KEY_KBDILLUMTOGGLE: return "KEY_KBDILLUMTOGGLE";
    case KEY_KBDILLUMDOWN: return "KEY_KBDILLUMDOWN";
    case KEY_KBDILLUMUP: return "KEY_KBDILLUMUP";
    case KEY_SEND: return "KEY_SEND";
    case KEY_REPLY: return "KEY_REPLY";
    case KEY_FORWARDMAIL: return "KEY_FORWARDMAIL";
    case KEY_SAVE: return "KEY_SAVE";
    case KEY_DOCUMENTS: return "KEY_DOCUMENTS";
    case KEY_BATTERY: return "KEY_BATTERY";
    case KEY_BLUETOOTH: return "KEY_BLUETOOTH";
    case KEY_WLAN: return "KEY_WLAN";
    case KEY_UWB: return "KEY_UWB";
    case KEY_UNKNOWN: return "KEY_UNKNOWN";
    case KEY_VIDEO_NEXT: return "KEY_VIDEO_NEXT";
    case KEY_VIDEO_PREV: return "KEY_VIDEO_PREV";
    case KEY_BRIGHTNESS_CYCLE: return "KEY_BRIGHTNESS_CYCLE";
    case KEY_BRIGHTNESS_AUTO: return "KEY_BRIGHTNESS_AUTO";
    case KEY_DISPLAY_OFF: return "KEY_DISPLAY_OFF";
    case KEY_WWAN: return "KEY_WWAN";
    case KEY_RFKILL: return "KEY_RFKILL";
    case KEY_MICMUTE: return "KEY_MICMUTE";

    // Misc
    case BTN_0: return "BTN_0";
    case BTN_1: return "BTN_1";
    case BTN_2: return "BTN_2";
    case BTN_3: return "BTN_3";
    case BTN_4: return "BTN_4";
    case BTN_5: return "BTN_5";
    case BTN_6: return "BTN_6";
    case BTN_7: return "BTN_7";
    case BTN_8: return "BTN_8";
    case BTN_9: return "BTN_9";

    // Mouse
    case BTN_LEFT: return "BTN_LEFT";
    case BTN_RIGHT: return "BTN_RIGHT";
    case BTN_MIDDLE: return "BTN_MIDDLE";
    case BTN_SIDE: return "BTN_SIDE";
    case BTN_EXTRA: return "BTN_EXTRA";
    case BTN_FORWARD: return "BTN_FORWARD";
    case BTN_BACK: return "BTN_BACK";
    case BTN_TASK: return "BTN_TASK";
    
    // Joystick
    case BTN_TRIGGER: return "BTN_TRIGGER";
    case BTN_THUMB: return "BTN_THUMB";
    case BTN_THUMB2: return "BTN_THUMB2";
    case BTN_TOP: return "BTN_TOP";
    case BTN_TOP2: return "BTN_TOP2";
    case BTN_PINKIE: return "BTN_PINKIE";
    case BTN_BASE: return "BTN_BASE";
    case BTN_BASE2: return "BTN_BASE2";
    case BTN_BASE3: return "BTN_BASE3";
    case BTN_BASE4: return "BTN_BASE4";
    case BTN_BASE5: return "BTN_BASE5";
    case BTN_BASE6: return "BTN_BASE6";
    case BTN_DEAD: return "BTN_DEAD";

    // Gamepad
    case BTN_SOUTH: return "BTN_SOUTH";
    case BTN_EAST: return "BTN_EAST";
    case BTN_C: return "BTN_C";
    case BTN_NORTH: return "BTN_NORTH";
    case BTN_WEST: return "BTN_WEST";
    case BTN_Z: return "BTN_Z";
    case BTN_TL: return "BTN_TL";
    case BTN_TR: return "BTN_TR";
    case BTN_TL2: return "BTN_TL2";
    case BTN_TR2: return "BTN_TR2";
    case BTN_SELECT: return "BTN_SELECT";
    case BTN_START: return "BTN_START";
    case BTN_MODE: return "BTN_MODE";
    case BTN_THUMBL: return "BTN_THUMBL";
    case BTN_THUMBR: return "BTN_THUMBR";

    // Digitizer
    case BTN_TOOL_PEN: return "BTN_TOOL_PEN";
    case BTN_TOOL_RUBBER: return "BTN_TOOL_RUBBER";
    case BTN_TOOL_BRUSH: return "BTN_TOOL_BRUSH";
    case BTN_TOOL_PENCIL: return "BTN_TOOL_PENCIL";
    case BTN_TOOL_AIRBRUSH: return "BTN_TOOL_AIRBRUSH";
    case BTN_TOOL_FINGER: return "BTN_TOOL_FINGER";
    case BTN_TOOL_MOUSE: return "BTN_TOOL_MOUSE";
    case BTN_TOOL_LENS: return "BTN_TOOL_LENS";
    case BTN_TOOL_QUINTTAP: return "BTN_TOOL_QUINTTAP";
    case BTN_STYLUS3: return "BTN_STYLUS3";
    case BTN_TOUCH: return "BTN_TOUCH";
    case BTN_STYLUS: return "BTN_STYLUS";
    case BTN_STYLUS2: return "BTN_STYLUS2";
    case BTN_TOOL_DOUBLETAP: return "BTN_TOOL_DOUBLETAP";
    case BTN_TOOL_TRIPLETAP: return "BTN_TOOL_TRIPLETAP";
    case BTN_TOOL_QUADTAP: return "BTN_TOOL_QUADTAP";

    // Wheel
    case BTN_GEAR_DOWN: return "BTN_GEAR_DOWN";
    case BTN_GEAR_UP: return "BTN_GEAR_UP";

    case KEY_OK: return "KEY_OK";
    case KEY_SELECT: return "KEY_SELECT";
    case KEY_GOTO: return "KEY_GOTO";
    case KEY_CLEAR: return "KEY_CLEAR";
    case KEY_POWER2: return "KEY_POWER2";
    case KEY_OPTION: return "KEY_OPTION";
    case KEY_INFO: return "KEY_INFO";
    case KEY_TIME: return "KEY_TIME";
    case KEY_VENDOR: return "KEY_VENDOR";
    case KEY_ARCHIVE: return "KEY_ARCHIVE";
    case KEY_PROGRAM: return "KEY_PROGRAM";
    case KEY_CHANNEL: return "KEY_CHANNEL";
    case KEY_FAVORITES: return "KEY_FAVORITES";
    case KEY_EPG: return "KEY_EPG";
    case KEY_PVR: return "KEY_PVR";
    case KEY_MHP: return "KEY_MHP";
    case KEY_LANGUAGE: return "KEY_LANGUAGE";
    case KEY_TITLE: return "KEY_TITLE";
    case KEY_SUBTITLE: return "KEY_SUBTITLE";
    case KEY_ANGLE: return "KEY_ANGLE";
    case KEY_FULL_SCREEN: return "KEY_FULL_SCREEN";
    case KEY_MODE: return "KEY_MODE";
    case KEY_KEYBOARD: return "KEY_KEYBOARD";
    case KEY_ASPECT_RATIO: return "KEY_ASPECT_RATIO";
    case KEY_PC: return "KEY_PC";
    case KEY_TV: return "KEY_TV";
    case KEY_TV2: return "KEY_TV2";
    case KEY_VCR: return "KEY_VCR";
    case KEY_VCR2: return "KEY_VCR2";
    case KEY_SAT: return "KEY_SAT";
    case KEY_SAT2: return "KEY_SAT2";
    case KEY_CD: return "KEY_CD";
    case KEY_TAPE: return "KEY_TAPE";
    case KEY_RADIO: return "KEY_RADIO";
    case KEY_TUNER: return "KEY_TUNER";
    case KEY_PLAYER: return "KEY_PLAYER";
    case KEY_TEXT: return "KEY_TEXT";
    case KEY_DVD: return "KEY_DVD";
    case KEY_AUX: return "KEY_AUX";
    case KEY_MP3: return "KEY_MP3";
    case KEY_AUDIO: return "KEY_AUDIO";
    case KEY_VIDEO: return "KEY_VIDEO";
    case KEY_DIRECTORY: return "KEY_DIRECTORY";
    case KEY_LIST: return "KEY_LIST";
    case KEY_MEMO: return "KEY_MEMO";
    case KEY_CALENDAR: return "KEY_CALENDAR";
    case KEY_RED: return "KEY_RED";
    case KEY_GREEN: return "KEY_GREEN";
    case KEY_YELLOW: return "KEY_YELLOW";
    case KEY_BLUE: return "KEY_BLUE";
    case KEY_CHANNELUP: return "KEY_CHANNELUP";
    case KEY_CHANNELDOWN: return "KEY_CHANNELDOWN";
    case KEY_FIRST: return "KEY_FIRST";
    case KEY_LAST: return "KEY_LAST";
    case KEY_AB: return "KEY_AB";
    case KEY_NEXT: return "KEY_NEXT";
    case KEY_RESTART: return "KEY_RESTART";
    case KEY_SLOW: return "KEY_SLOW";
    case KEY_SHUFFLE: return "KEY_SHUFFLE";
    case KEY_BREAK: return "KEY_BREAK";
    case KEY_PREVIOUS: return "KEY_PREVIOUS";
    case KEY_DIGITS: return "KEY_DIGITS";
    case KEY_TEEN: return "KEY_TEEN";
    case KEY_TWEN: return "KEY_TWEN";
    case KEY_VIDEOPHONE: return "KEY_VIDEOPHONE";
    case KEY_GAMES: return "KEY_GAMES";
    case KEY_ZOOMIN: return "KEY_ZOOMIN";
    case KEY_ZOOMOUT: return "KEY_ZOOMOUT";
    case KEY_ZOOMRESET: return "KEY_ZOOMRESET";
    case KEY_WORDPROCESSOR: return "KEY_WORDPROCESSOR";
    case KEY_EDITOR: return "KEY_EDITOR";
    case KEY_SPREADSHEET: return "KEY_SPREADSHEET";
    case KEY_GRAPHICSEDITOR: return "KEY_GRAPHICSEDITOR";
    case KEY_PRESENTATION: return "KEY_PRESENTATION";
    case KEY_DATABASE: return "KEY_DATABASE";
    case KEY_NEWS: return "KEY_NEWS";
    case KEY_VOICEMAIL: return "KEY_VOICEMAIL";
    case KEY_ADDRESSBOOK: return "KEY_ADDRESSBOOK";
    case KEY_MESSENGER: return "KEY_MESSENGER";
    case KEY_DISPLAYTOGGLE: return "KEY_DISPLAYTOGGLE";
    case KEY_SPELLCHECK: return "KEY_SPELLCHECK";
    case KEY_LOGOFF: return "KEY_LOGOFF";
    case KEY_DOLLAR: return "KEY_DOLLAR";
    case KEY_EURO: return "KEY_EURO";
    case KEY_FRAMEBACK: return "KEY_FRAMEBACK";
    case KEY_FRAMEFORWARD: return "KEY_FRAMEFORWARD";
    case KEY_CONTEXT_MENU: return "KEY_CONTEXT_MENU";
    case KEY_MEDIA_REPEAT: return "KEY_MEDIA_REPEAT";
    case KEY_10CHANNELSUP: return "KEY_10CHANNELSUP";
    case KEY_10CHANNELSDOWN: return "KEY_10CHANNELSDOWN";
    case KEY_IMAGES: return "KEY_IMAGES";
    case KEY_DEL_EOL: return "KEY_DEL_EOL";
    case KEY_DEL_EOS: return "KEY_DEL_EOS";
    case KEY_INS_LINE: return "KEY_INS_LINE";
    case KEY_DEL_LINE: return "KEY_DEL_LINE";
    case KEY_FN: return "KEY_FN";
    case KEY_FN_ESC: return "KEY_FN_ESC";
    case KEY_FN_F1: return "KEY_FN_F1";
    case KEY_FN_F2: return "KEY_FN_F2";
    case KEY_FN_F3: return "KEY_FN_F3";
    case KEY_FN_F4: return "KEY_FN_F4";
    case KEY_FN_F5: return "KEY_FN_F5";
    case KEY_FN_F6: return "KEY_FN_F6";
    case KEY_FN_F7: return "KEY_FN_F7";
    case KEY_FN_F8: return "KEY_FN_F8";
    case KEY_FN_F9: return "KEY_FN_F9";
    case KEY_FN_F10: return "KEY_FN_F10";
    case KEY_FN_F11: return "KEY_FN_F11";
    case KEY_FN_F12: return "KEY_FN_F12";
    case KEY_FN_1: return "KEY_FN_1";
    case KEY_FN_2: return "KEY_FN_2";
    case KEY_FN_D: return "KEY_FN_D";
    case KEY_FN_E: return "KEY_FN_E";
    case KEY_FN_F: return "KEY_FN_F";
    case KEY_FN_S: return "KEY_FN_S";
    case KEY_FN_B: return "KEY_FN_B";
    case KEY_BRL_DOT1: return "KEY_BRL_DOT1";
    case KEY_BRL_DOT2: return "KEY_BRL_DOT2";
    case KEY_BRL_DOT3: return "KEY_BRL_DOT3";
    case KEY_BRL_DOT4: return "KEY_BRL_DOT4";
    case KEY_BRL_DOT5: return "KEY_BRL_DOT5";
    case KEY_BRL_DOT6: return "KEY_BRL_DOT6";
    case KEY_BRL_DOT7: return "KEY_BRL_DOT7";
    case KEY_BRL_DOT8: return "KEY_BRL_DOT8";
    case KEY_BRL_DOT9: return "KEY_BRL_DOT9";
    case KEY_BRL_DOT10: return "KEY_BRL_DOT10";
    case KEY_NUMERIC_0: return "KEY_NUMERIC_0";
    case KEY_NUMERIC_1: return "KEY_NUMERIC_1";
    case KEY_NUMERIC_2: return "KEY_NUMERIC_2";
    case KEY_NUMERIC_3: return "KEY_NUMERIC_3";
    case KEY_NUMERIC_4: return "KEY_NUMERIC_4";
    case KEY_NUMERIC_5: return "KEY_NUMERIC_5";
    case KEY_NUMERIC_6: return "KEY_NUMERIC_6";
    case KEY_NUMERIC_7: return "KEY_NUMERIC_7";
    case KEY_NUMERIC_8: return "KEY_NUMERIC_8";
    case KEY_NUMERIC_9: return "KEY_NUMERIC_9";
    case KEY_NUMERIC_STAR: return "KEY_NUMERIC_STAR";
    case KEY_NUMERIC_POUND: return "KEY_NUMERIC_POUND";
    case KEY_NUMERIC_A: return "KEY_NUMERIC_A";
    case KEY_NUMERIC_B: return "KEY_NUMERIC_B";
    case KEY_NUMERIC_C: return "KEY_NUMERIC_C";
    case KEY_NUMERIC_D: return "KEY_NUMERIC_D";
    case KEY_CAMERA_FOCUS: return "KEY_CAMERA_FOCUS";
    case KEY_WPS_BUTTON: return "KEY_WPS_BUTTON";
    case KEY_TOUCHPAD_TOGGLE: return "KEY_TOUCHPAD_TOGGLE";
    case KEY_TOUCHPAD_ON: return "KEY_TOUCHPAD_ON";
    case KEY_TOUCHPAD_OFF: return "KEY_TOUCHPAD_OFF";
    case KEY_CAMERA_ZOOMIN: return "KEY_CAMERA_ZOOMIN";
    case KEY_CAMERA_ZOOMOUT: return "KEY_CAMERA_ZOOMOUT";
    case KEY_CAMERA_UP: return "KEY_CAMERA_UP";
    case KEY_CAMERA_DOWN: return "KEY_CAMERA_DOWN";
    case KEY_CAMERA_LEFT: return "KEY_CAMERA_LEFT";
    case KEY_CAMERA_RIGHT: return "KEY_CAMERA_RIGHT";
    case KEY_ATTENDANT_ON: return "KEY_ATTENDANT_ON";
    case KEY_ATTENDANT_OFF: return "KEY_ATTENDANT_OFF";
    case KEY_ATTENDANT_TOGGLE: return "KEY_ATTENDANT_TOGGLE";
    case KEY_LIGHTS_TOGGLE: return "KEY_LIGHTS_TOGGLE";
    case BTN_DPAD_UP: return "BTN_DPAD_UP";
    case BTN_DPAD_DOWN: return "BTN_DPAD_DOWN";
    case BTN_DPAD_LEFT: return "BTN_DPAD_LEFT";
    case BTN_DPAD_RIGHT: return "BTN_DPAD_RIGHT";
    case KEY_ALS_TOGGLE: return "KEY_ALS_TOGGLE";
    case KEY_ROTATE_LOCK_TOGGLE: return "KEY_ROTATE_LOCK_TOGGLE";
    case KEY_BUTTONCONFIG: return "KEY_BUTTONCONFIG";
    case KEY_TASKMANAGER: return "KEY_TASKMANAGER";
    case KEY_JOURNAL: return "KEY_JOURNAL";
    case KEY_CONTROLPANEL: return "KEY_CONTROLPANEL";
    case KEY_APPSELECT: return "KEY_APPSELECT";
    case KEY_SCREENSAVER: return "KEY_SCREENSAVER";
    case KEY_VOICECOMMAND: return "KEY_VOICECOMMAND";
    case KEY_ASSISTANT: return "KEY_ASSISTANT";
    case KEY_KBD_LAYOUT_NEXT: return "KEY_KBD_LAYOUT_NEXT";
    case KEY_EMOJI_PICKER: return "KEY_EMOJI_PICKER";
    case KEY_DICTATE: return "KEY_DICTATE";
    case KEY_BRIGHTNESS_MIN: return "KEY_BRIGHTNESS_MIN";
    case KEY_BRIGHTNESS_MAX: return "KEY_BRIGHTNESS_MAX";
    case KEY_KBDINPUTASSIST_PREV: return "KEY_KBDINPUTASSIST_PREV";
    case KEY_KBDINPUTASSIST_NEXT: return "KEY_KBDINPUTASSIST_NEXT";
    case KEY_KBDINPUTASSIST_PREVGROUP: return "KEY_KBDINPUTASSIST_PREVGROUP";
    case KEY_KBDINPUTASSIST_NEXTGROUP: return "KEY_KBDINPUTASSIST_NEXTGROUP";
    case KEY_KBDINPUTASSIST_ACCEPT: return "KEY_KBDINPUTASSIST_ACCEPT";
    case KEY_KBDINPUTASSIST_CANCEL: return "KEY_KBDINPUTASSIST_CANCEL";
    case KEY_RIGHT_UP: return "KEY_RIGHT_UP";
    case KEY_RIGHT_DOWN: return "KEY_RIGHT_DOWN";
    case KEY_LEFT_UP: return "KEY_LEFT_UP";
    case KEY_LEFT_DOWN: return "KEY_LEFT_DOWN";
    case KEY_ROOT_MENU: return "KEY_ROOT_MENU";
    case KEY_MEDIA_TOP_MENU: return "KEY_MEDIA_TOP_MENU";
    case KEY_NUMERIC_11: return "KEY_NUMERIC_11";
    case KEY_NUMERIC_12: return "KEY_NUMERIC_12";
    case KEY_AUDIO_DESC: return "KEY_AUDIO_DESC";
    case KEY_3D_MODE: return "KEY_3D_MODE";
    case KEY_NEXT_FAVORITE: return "KEY_NEXT_FAVORITE";
    case KEY_STOP_RECORD: return "KEY_STOP_RECORD";
    case KEY_PAUSE_RECORD: return "KEY_PAUSE_RECORD";
    case KEY_VOD: return "KEY_VOD";
    case KEY_UNMUTE: return "KEY_UNMUTE";
    case KEY_FASTREVERSE: return "KEY_FASTREVERSE";
    case KEY_SLOWREVERSE: return "KEY_SLOWREVERSE";
    case KEY_DATA: return "KEY_DATA";
    case KEY_ONSCREEN_KEYBOARD: return "KEY_ONSCREEN_KEYBOARD";
    //case BTN_TRIGGER_HAPPY: return "BTN_TRIGGER_HAPPY";
    case BTN_TRIGGER_HAPPY1: return "BTN_TRIGGER_HAPPY1";
    case BTN_TRIGGER_HAPPY2: return "BTN_TRIGGER_HAPPY2";
    case BTN_TRIGGER_HAPPY3: return "BTN_TRIGGER_HAPPY3";
    case BTN_TRIGGER_HAPPY4: return "BTN_TRIGGER_HAPPY4";
    case BTN_TRIGGER_HAPPY5: return "BTN_TRIGGER_HAPPY5";
    case BTN_TRIGGER_HAPPY6: return "BTN_TRIGGER_HAPPY6";
    case BTN_TRIGGER_HAPPY7: return "BTN_TRIGGER_HAPPY7";
    case BTN_TRIGGER_HAPPY8: return "BTN_TRIGGER_HAPPY8";
    case BTN_TRIGGER_HAPPY9: return "BTN_TRIGGER_HAPPY9";
    case BTN_TRIGGER_HAPPY10: return "BTN_TRIGGER_HAPPY10";
    case BTN_TRIGGER_HAPPY11: return "BTN_TRIGGER_HAPPY11";
    case BTN_TRIGGER_HAPPY12: return "BTN_TRIGGER_HAPPY12";
    case BTN_TRIGGER_HAPPY13: return "BTN_TRIGGER_HAPPY13";
    case BTN_TRIGGER_HAPPY14: return "BTN_TRIGGER_HAPPY14";
    case BTN_TRIGGER_HAPPY15: return "BTN_TRIGGER_HAPPY15";
    case BTN_TRIGGER_HAPPY16: return "BTN_TRIGGER_HAPPY16";
    case BTN_TRIGGER_HAPPY17: return "BTN_TRIGGER_HAPPY17";
    case BTN_TRIGGER_HAPPY18: return "BTN_TRIGGER_HAPPY18";
    case BTN_TRIGGER_HAPPY19: return "BTN_TRIGGER_HAPPY19";
    case BTN_TRIGGER_HAPPY20: return "BTN_TRIGGER_HAPPY20";
    case BTN_TRIGGER_HAPPY21: return "BTN_TRIGGER_HAPPY21";
    case BTN_TRIGGER_HAPPY22: return "BTN_TRIGGER_HAPPY22";
    case BTN_TRIGGER_HAPPY23: return "BTN_TRIGGER_HAPPY23";
    case BTN_TRIGGER_HAPPY24: return "BTN_TRIGGER_HAPPY24";
    case BTN_TRIGGER_HAPPY25: return "BTN_TRIGGER_HAPPY25";
    case BTN_TRIGGER_HAPPY26: return "BTN_TRIGGER_HAPPY26";
    case BTN_TRIGGER_HAPPY27: return "BTN_TRIGGER_HAPPY27";
    case BTN_TRIGGER_HAPPY28: return "BTN_TRIGGER_HAPPY28";
    case BTN_TRIGGER_HAPPY29: return "BTN_TRIGGER_HAPPY29";
    case BTN_TRIGGER_HAPPY30: return "BTN_TRIGGER_HAPPY30";
    case BTN_TRIGGER_HAPPY31: return "BTN_TRIGGER_HAPPY31";
    case BTN_TRIGGER_HAPPY32: return "BTN_TRIGGER_HAPPY32";
    case BTN_TRIGGER_HAPPY33: return "BTN_TRIGGER_HAPPY33";
    case BTN_TRIGGER_HAPPY34: return "BTN_TRIGGER_HAPPY34";
    case BTN_TRIGGER_HAPPY35: return "BTN_TRIGGER_HAPPY35";
    case BTN_TRIGGER_HAPPY36: return "BTN_TRIGGER_HAPPY36";
    case BTN_TRIGGER_HAPPY37: return "BTN_TRIGGER_HAPPY37";
    case BTN_TRIGGER_HAPPY38: return "BTN_TRIGGER_HAPPY38";
    case BTN_TRIGGER_HAPPY39: return "BTN_TRIGGER_HAPPY39";
    case BTN_TRIGGER_HAPPY40: return "BTN_TRIGGER_HAPPY40";
    case KEY_MAX: return "KEY_MAX";
/* Relative axes
    case REL_X: return "REL_X";
    case REL_Y: return "REL_Y";
    case REL_Z: return "REL_Z";
    case REL_RX: return "REL_RX";
    case REL_RY: return "REL_RY";
    case REL_RZ: return "REL_RZ";
    case REL_HWHEEL: return "REL_HWHEEL";
    case REL_DIAL: return "REL_DIAL";
    case REL_WHEEL: return "REL_WHEEL";
    case REL_MISC: return "REL_MISC";
    case REL_RESERVED: return "REL_RESERVED";
    case REL_WHEEL_HI_RES: return "REL_WHEEL_HI_RES";
    case REL_HWHEEL_HI_RES: return "REL_HWHEEL_HI_RES";
    case REL_MAX: return "REL_MAX";
*/
/* Absolute axes
    case ABS_X: return "ABS_X";
    case ABS_Y: return "ABS_Y";
    case ABS_Z: return "ABS_Z";
    case ABS_RX: return "ABS_RX";
    case ABS_RY: return "ABS_RY";
    case ABS_RZ: return "ABS_RZ";
    case ABS_THROTTLE: return "ABS_THROTTLE";
    case ABS_RUDDER: return "ABS_RUDDER";
    case ABS_WHEEL: return "ABS_WHEEL";
    case ABS_GAS: return "ABS_GAS";
    case ABS_BRAKE: return "ABS_BRAKE";
    case ABS_HAT0X: return "ABS_HAT0X";
    case ABS_HAT0Y: return "ABS_HAT0Y";
    case ABS_HAT1X: return "ABS_HAT1X";
    case ABS_HAT1Y: return "ABS_HAT1Y";
    case ABS_HAT2X: return "ABS_HAT2X";
    case ABS_HAT2Y: return "ABS_HAT2Y";
    case ABS_HAT3X: return "ABS_HAT3X";
    case ABS_HAT3Y: return "ABS_HAT3Y";
    case ABS_PRESSURE: return "ABS_PRESSURE";
    case ABS_DISTANCE: return "ABS_DISTANCE";
    case ABS_TILT_X: return "ABS_TILT_X";
    case ABS_TILT_Y: return "ABS_TILT_Y";
    case ABS_TOOL_WIDTH: return "ABS_TOOL_WIDTH";
    case ABS_VOLUME: return "ABS_VOLUME";
    case ABS_MISC: return "ABS_MISC";
    case ABS_RESERVED: return "ABS_RESERVED";
    case ABS_MT_SLOT: return "ABS_MT_SLOT";
    case ABS_MT_TOUCH_MAJOR: return "ABS_MT_TOUCH_MAJOR";
    case ABS_MT_TOUCH_MINOR: return "ABS_MT_TOUCH_MINOR";
    case ABS_MT_WIDTH_MAJOR: return "ABS_MT_WIDTH_MAJOR";
    case ABS_MT_WIDTH_MINOR: return "ABS_MT_WIDTH_MINOR";
    case ABS_MT_ORIENTATION: return "ABS_MT_ORIENTATION";
    case ABS_MT_POSITION_X: return "ABS_MT_POSITION_X";
    case ABS_MT_POSITION_Y: return "ABS_MT_POSITION_Y";
    case ABS_MT_TOOL_TYPE: return "ABS_MT_TOOL_TYPE";
    case ABS_MT_BLOB_ID: return "ABS_MT_BLOB_ID";
    case ABS_MT_TRACKING_ID: return "ABS_MT_TRACKING_ID";
    case ABS_MT_PRESSURE: return "ABS_MT_PRESSURE";
    case ABS_MT_DISTANCE: return "ABS_MT_DISTANCE";
    case ABS_MT_TOOL_X: return "ABS_MT_TOOL_X";
    case ABS_MT_TOOL_Y: return "ABS_MT_TOOL_Y";
    case ABS_MAX: return "ABS_MAX";
*/
/* Switch events
    case SW_LID: return "SW_LID";
    case SW_TABLET_MODE: return "SW_TABLET_MODE";
    case SW_HEADPHONE_INSERT: return "SW_HEADPHONE_INSERT";
    case SW_RFKILL_ALL: return "SW_RFKILL_ALL";
    case SW_MICROPHONE_INSERT: return "SW_MICROPHONE_INSERT";
    case SW_DOCK: return "SW_DOCK";
    case SW_LINEOUT_INSERT: return "SW_LINEOUT_INSERT";
    case SW_JACK_PHYSICAL_INSERT: return "SW_JACK_PHYSICAL_INSERT";
    case SW_VIDEOOUT_INSERT: return "SW_VIDEOOUT_INSERT";
    case SW_CAMERA_LENS_COVER: return "SW_CAMERA_LENS_COVER";
    case SW_KEYPAD_SLIDE: return "SW_KEYPAD_SLIDE";
    case SW_FRONT_PROXIMITY: return "SW_FRONT_PROXIMITY";
    case SW_ROTATE_LOCK: return "SW_ROTATE_LOCK";
    case SW_LINEIN_INSERT: return "SW_LINEIN_INSERT";
    case SW_MUTE_DEVICE: return "SW_MUTE_DEVICE";
    case SW_PEN_INSERTED: return "SW_PEN_INSERTED";
    case SW_MACHINE_COVER: return "SW_MACHINE_COVER";
    case SW_MAX: return "SW_MAX";
*/
/* Misc events
    case MSC_SERIAL: return "MSC_SERIAL";
    case MSC_PULSELED: return "MSC_PULSELED";
    case MSC_GESTURE: return "MSC_GESTURE";
    case MSC_RAW: return "MSC_RAW";
    case MSC_SCAN: return "MSC_SCAN";
    case MSC_TIMESTAMP: return "MSC_TIMESTAMP";
    case MSC_MAX: return "MSC_MAX";
*/
/* LEDs
    case LED_NUML: return "LED_NUML";
    case LED_CAPSL: return "LED_CAPSL";
    case LED_SCROLLL: return "LED_SCROLLL";
    case LED_COMPOSE: return "LED_COMPOSE";
    case LED_KANA: return "LED_KANA";
    case LED_SLEEP: return "LED_SLEEP";
    case LED_SUSPEND: return "LED_SUSPEND";
    case LED_MUTE: return "LED_MUTE";
    case LED_MISC: return "LED_MISC";
    case LED_MAIL: return "LED_MAIL";
    case LED_CHARGING: return "LED_CHARGING";
    case LED_MAX: return "LED_MAX";
*/
    default:
      break;
  }
  snprintf(buf, 64, "unknown 0x%x / %d", v, v);
  buf[63] = '\0';
  return buf;
}

