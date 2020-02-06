#pragma once


enum EButton {
  kButtonUp,
  kButtonDown,
  kButtonLeft,
  kButtonRight,

  kButtonPower,
  kButtonHome,
  kButtonMenu,
  kButtonBack,
  kButtonOK,

  kButtonCount
};

// Returns true once per button press.
// (pressing the button sets a latch, calling testButton() releases that latch)
bool testButton(EButton);

void startInputListenerThread();
