#!/bin/bash

# Lock OFA clocks to max
echo 1 > /sys/kernel/debug/bpmp/debug/clk/ofa/mrq_rate_locked
cat /sys/kernel/debug/bpmp/debug/clk/ofa/max_rate > /sys/kernel/debug/bpmp/debug/clk/ofa/rate

# Pin camera RTCPU on
# This works around a (firmware?) bug in L4T R36, where the cameras only work once per boot.
# The bug is only triggered when the RTCPU goes into suspend; pinning its power on prevents suspend mode.

echo on > /sys/devices/platform/*rtcpu/power/control

