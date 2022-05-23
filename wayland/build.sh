#!/bin/bash
wayland-scanner client-header fullscreen-shell-unstable-v1.xml fullscreen-shell-unstable-v1-client-header.h
wayland-scanner private-code fullscreen-shell-unstable-v1.xml fullscreen-shell-unstable-v1-protocol.c

