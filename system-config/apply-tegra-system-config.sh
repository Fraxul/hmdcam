#!/bin/bash

if [ ! -f /etc/nv_tegra_release ]; then
  echo "This script is only meant to be run on a Linux for Tegra embedded system, not on your host development machine."
  exit 1
fi

if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root."
  exit 1
fi

# Set cwd to the directory containing the apply-system-config script
cd "$(dirname "${BASH_SOURCE[0]}")"

APP_USER=$(id -nu 1000)
echo -e "Assuming the user that will be running the app is \"${APP_USER}\""

# Add user to the plugdev and input groups
# plugdev is required for HMD access
# input is required for raw input device (kb/m, or remotes that emulate those) access
usermod -aG plugdev ${APP_USER}
usermod -aG input ${APP_USER}

# Disable graphical environment at boot
if [ "$(systemctl get-default)" != "multi-user.target" ]; then
  echo "Setting systemd default target to multi-user.target"
  echo "Note: if you need to reenable the graphical environment, run:"
  echo "  systemctl set-default graphical.target   # Default to running graphical environment on boot"
  echo "  systemctl isolate graphical.target       # Start graphical environment immediately"

  systemctl isolate multi-user.target
  systemctl set-default multi-user.target
fi

# Disable some unwanted services
systemctl disable nvzramconfig.service
systemctl disable nvmemwarning.service

# Put /tmp on tmpfs to reduce nvme writes
systemctl enable /usr/share/systemd/tmp.mount

# Install updated nvargus-daemon service script
diff -q assets/nvargus-daemon.service /etc/systemd/system/nvargus-daemon.service >/dev/null
if [ $? -ne 0 ]; then
  echo "Updating nvargus-daemon.service"
  cp assets/nvargus-daemon.service /etc/systemd/system/
fi

# Install power management override script
cp assets/set-power-management-overrides.sh /etc/systemd/
cp assets/set-power-management-overrides.service /etc/systemd/system/
systemctl enable set-power-management-overrides.service

# Install CAN bus enable script
# (this is a lot simpler than reconfiguring NetworkManager to run it for us)
cp assets/canbus.service /etc/systemd/system/
systemctl enable canbus.service

# Apply udev rules
cp assets/83-hmd.rules /etc/udev/rules.d/
udevadm control --reload
udevadm trigger

# Load required modules
diff -q assets/modules /etc/modules >/dev/null
if [ $? -ne 0 ]; then
  echo "Updating /etc/modules -- original will be saved as /etc/modules.bak"
  echo "You will probably need to reboot to ensure that modules are loaded correctly."
  mv /etc/modules /etc/modules.bak
  cp assets/modules
fi

# Apply logind config to ignore power button presses
# This prevents accidental poweroff via USB-HID remote with power button
if [ ! -f /etc/systemd/logind.conf.d/ignore-power-button.conf ]; then
  echo "Updating systemd-logind config. This will require a reboot to take effect."
  echo "Or you can run 'sudo systemctl restart systemd-logind', but this will kill your login session."
  mkdir -p /etc/systemd/logind.conf.d
  cp assets/ignore-power-button.conf /etc/systemd/logind.conf.d/
fi

# Install camera_overrides file if one isn't already present.
if [ ! -f /var/nvidia/nvcam/settings/camera_overrides.isp ]; then
  echo "Installing default camera_overrides.isp file (based on IMX290 sensor tuning)"
  cp assets/camera_overrides-li-imx290.isp /var/nvidia/nvcam/settings/camera_overrides.isp
fi

# Final step: nvpmodel should be set to MAXN
nvpmodel -q | grep -q MAXN
if [ $? -ne 0 ]; then
  echo ""
  echo "*** Invoking nvpmodel to set the power mode to MAXN. This will ask you to reboot."
  echo "*** Please respond 'yes' here, since the setup script is now complete."
  echo ""
  nvpmodel -m 0
fi

