#!/bin/bash
# Displays a raw CSI camera stream full-screen on an attached display via DRM (no X-server running)
# Useful for setting focus on cameras.
#
# Requires the nvidia-l4t-gstreamer package to be installed.
#

usage() {
  echo "Usage: gst-camera-to-display.sh sensor_id [sensor_mode]"
  exit 1
}

if [ -z "$1" ]; then
  echo "Error: No sensor ID specified."
  usage
fi

sensor_mode=1
if [ ! -z "$2" ]; then
  sensor_mode=$2
fi


gst-launch-1.0 nvarguscamerasrc sensor_id=$1 sensor_mode=${sensor_mode} ! 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)60/1' ! nvdrmvideosink -e
