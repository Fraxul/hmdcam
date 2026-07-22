#!/bin/bash
set -e

CLK=/sys/kernel/debug/bpmp/debug/clk

lock_max() {
  local name=$1
  local max
  max=$(cat "$CLK/$name/max_rate")
  echo 1 > "$CLK/$name/mrq_rate_locked"
  echo "$max" > "$CLK/$name/rate"
  echo "$name locked to $(cat "$CLK/$name/rate") Hz"
}

# Lock OFA, ISP, and VI clocks to maximum.
# Max OFA is required to consistently hit DepthMapGeneratorOFA deadlines with no jitter.
# Max ISP and VI is required to avoid introducing wakeup jitter in ArgusCamera::readFrame()

lock_max isp
lock_max vi
lock_max ofa

# Pin camera RTCPU on
# This works around a (firmware?) bug in L4T R36, where the cameras only work once per boot.
# The bug is only triggered when the RTCPU goes into suspend; pinning its power on prevents suspend mode.

echo on > /sys/devices/platform/*rtcpu/power/control

