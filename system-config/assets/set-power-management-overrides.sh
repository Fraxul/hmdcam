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

# Disable GPU engine-level power gating (ELPG).
# The GPU clock is locked to a fixed rate to avoid DVFS transition latency, but ELPG
# is an orthogonal knob: it power-gates the GR engine during brief idle windows. Each
# un-gate forces the GPC NAFLL to relock from its ~115MHz reference back up to the locked
# rate, parking the graphics clock at base for ~800us right as GPU work resumes -- which
# shows up as per-frame pacing jitter. Note: devfreq cur_freq keeps reading the locked
# rate throughout, so this is only visible in nsys GPU-metrics, not tegrastats.
# Disabling ELPG (and adaptive ELPG) keeps the GR engine powered so the clock stays pinned.
#
# ELPG lock can be verified by reading the /sys/kernel/debug/gpu.0/elpg_transitions counter.
# With ELPG disabled, the counter should stay fixed.
#

GPU=/sys/devices/platform/17000000.gpu
echo 0 > "$GPU/elpg_enable"
echo 0 > "$GPU/aelpg_enable"
echo "ELPG disabled (elpg_enable=$(cat $GPU/elpg_enable) aelpg_enable=$(cat $GPU/aelpg_enable))"

