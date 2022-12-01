# hmdcam: An indirect vision system for fursuit pilots


This application is designed for camera passthrough to a head-mounted display.
It runs on the Nvidia Jetson family of SoCs and has been developed/tested against the Lenovo Explorer, HP VR1000, and HTC Vive, but should work on any HMD supported by Monado.

Features:
- Distortion-corrected monoscopic and stereoscopic views positioned in 3d space
- Depth-mapped stereoscopic views to support off-axis viewing and multiview stitching
  - Both horizontal (left-right) and vertical (top-bottom) stereo pairs are supported
  - Depending on your camera configuration and platform, this may require an additional discrete GPU or VPU for processing.
  - Supports multiple depth processing backends:
    - Nvidia VPI 2.0 on the Jetson's integrated GPU (requires JetPack 5.0.2+, Orin or Xavier modules only)
    - OpenCV+CUDA on a separate Nvidia discrete GPU
    - Luxonis DepthAI VPU modules
- Configuration menu with built-in calibration tools
- Remote viewing via RTSP (embedded live555 server + nvenc)
- Remote debugging via RDMA-over-Infiniband (`rdma-client` binary)

Important repository structure:
| Path           | Description  |
| -------------- | ------------ |
| hmdcam         |  Main application -- runs on the Jetson |
| rdma-client    | Remote-debugging application; streams uncompressed framebuffers from `hmdcam` using RDMA over Infiniband. |
| dgpu-worker    | Stereo disparity computation worker for CUDA discrete GPUs. Runs under `hmdcam` or `rdma-client` |
| depthai-worker | Stereo disparity computation worker for Luxonis DepthAI VPUs. Runs under `hmdcam` or `rdma-client` |
| dgpu-fans      | Simple daemon to control a dGPU fan connected to the Jetson's PWM interface |
| rhi            | Render Hardware Interface -- wrappers over OpenGL. |
| common         | Library functions (mostly camera related) |
| rdma           | RDMA library/framework |

Requirements:
- A Jetson board with one or more CSI cameras
  - The cameras should be capable of capturing at the display rate of your HMD (90fps for the Vive or Explorer)
  - 4-lane IMX290 sensors are preferred: they can capture 1920x1080 at up to 120fps
  - 2-lane IMX219 sensors are a good fallback option, capturing 1280x720 at up to 120fps
- A head-mounted display supported by Monado
  - First-generation Windows Mixed Reality HMDs are excellent candidates: they are inexpensive, compact, and have reasonably good optics and displays.
  - A stripped-down WMR HMD can be made even smaller by replacing its captive cable with [this breakout board](https://github.com/Fraxul/Explorer-Breakout)
- (Optional) an Nvidia discrete GPU or Luxonis DepthAI VPUs for stereo processing
  - A GTX1050 MXM3 module can handle 2 stereo pairs with 1:4 sampling (480x270 depth resolution) at 90+ FPS, but consumes ~40 watts.
  - One VPU per stereo pair with 1:4 sampling (480x270 depth resolution). Each VPU consumes about 2 watts (tested using [OAK-FFC-3P](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM1090.html) modules without attached cameras).

The reference test/development platform is:
- Jetson AGX Xavier devkit running L4T r32.2.1
- eConSystems e-CAM20\_CUXVR camera kit: quad 4-lane IMX290 sensors, 1920x1080 at up to 120fps. Arranged as two 90°-rotated horizontal stereo pairs.
- Lenovo Explorer HMD: 2880x1440 at 90fps.
- Mellanox ConnectX-3 FDR Infiniband card (CX353A-FCBT) for remote debugging

Future development will move to the Jetson AGX Orin and L4T r35.

