# hmdcam: An indirect vision system for fursuit pilots


This application is designed for camera passthrough to a head-mounted display.
It runs on the Nvidia Jetson family of SoCs and has been developed/tested against the HTC Vive (but should work on any HMD supported by Monado).

Features:
- Distortion-corrected monoscopic and stereoscopic views positioned in 3d space
- Depth-mapped stereoscopic views to support off-axis viewing and multiview stitching
  - This requires an additional discrete GPU for processing, since the Jetson iGPU isn't powerful enough.
- Configuration menu with built-in calibration tools
- Remote viewing via RTSP (embedded live555 server + nvenc)
- Remote debugging via RDMA-over-Infiniband (`rdma-client` binary)

Important repository structure:
| Path        | Description  |
| ----------- | ------------ |
| hmdcam      |  Main application -- runs on the Jetson |
| rdma-client | Remote-debugging application; streams uncompressed framebuffers from `hmdcam` using RDMA over Infiniband. |
| dgpu-worker | Stereo disparity computation worker. Runs under `hmdcam` or `rdma-client` |
| dgpu-fans   | Simple daemon to control a dGPU fan connected to the Jetson's PWM interface |
| rhi         | Render Hardware Interface -- wrappers over OpenGL. |
| common      | Library functions (mostly camera related) |
| rdma        | RDMA library/framework |

Requirements:
- A Jetson board with one or more CSI cameras
  - The cameras should be capable of capturing at the display rate of your HMD (90fps for the Vive)
  - 4-lane IMX290 sensors are preferred: they can capture 1920x1080 at up to 120fps
  - 2-lane IMX219 sensors are a good fallback option, capturing 1280x720 at up to 120fps
- A head-mounted display supported by Monado
- (Optional) an Nvidia discrete GPU for stereo processing
  - A GTX1050 MXM3 module can handle 1 stereo pair at 90+ FPS
