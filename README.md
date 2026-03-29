# yolov10_ros — NCAT AutoDrive Perception Stack

A ROS 1 (Noetic) autonomous driving perception pipeline built on YOLOv10, EfficientNet B0 classifiers, TwinLiteNet lane segmentation, and LiDAR-camera fusion. Developed at North Carolina A&T State University for the AutoDrive Challenge III competition.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Node Descriptions](#node-descriptions)
4. [ROS Topics](#ros-topics)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Launch Files](#launch-files)
8. [Testing with a ROS Bag](#testing-with-a-ros-bag)
9. [Scoring CAN Node](#scoring-can-node)
10. [Configuration Parameters](#configuration-parameters)
11. [Calibration Files](#calibration-files)
12. [Directory Structure](#directory-structure)
13. [Troubleshooting](#troubleshooting)

---

## Overview

This package provides a full perception stack that:

- Detects objects (cars, pedestrians, cyclists, traffic lights, traffic signs) using **YOLOv10** trained on BDD100K
- Classifies traffic sign types (stop, yield, speed limit, etc.) and traffic light states (red/green/yellow) using **EfficientNet B0**
- Reads speed limit values from signs using **EasyOCR**
- Segments drivable area and lane lines using **TwinLiteNet**
- Fuses camera detections with **LiDAR point clouds** to estimate object distance and position
- Publishes all results to `perception_msgs` topics consumed by the downstream autonomy stack

---

## System Architecture

```
Camera ──→ [detector_node] ──→ /yolov10/detections ──→ [light_classifier_node]
                │                                   ──→ [sign_classifier_node] ──→ [speed_limit_ocr_node]
                │                                   ──→ [lane_seg_node]
                ↓
         /yolov10/image_raw
                │
LiDAR ──→ [lidar_stamp_relay] ──→ /lidar/points_corrected
                │
                ↓
         [perception_fusion_node]
                │
                ├──→ /objects_scoring              (perception_msgs/Objects)
                ├──→ /TrafficSigns_scoring          (perception_msgs/TrafficSigns)
                ├──→ /TrafficSignalsHead_scoring    (perception_msgs/TrafficSignalsHead)
                ├──→ /LaneMarkings_scoring          (perception_msgs/LaneMarkings)
                ├──→ /LimitLines_scoring            (perception_msgs/LimitLines)
                └──→ /road_state_scoring            (perception_msgs/road_state)

All outputs ──→ [perception_viz_node] ──→ /perception/annotated

Scoring topics ──→ [scoring_can] (terminal monitor)
```

---

## Node Descriptions

| Node | Script | Purpose |
|---|---|---|
| `yolov10_detector` | `detector_node.py` | Runs YOLOv10 inference on camera frames; publishes bounding-box `DetectionArray` and decoded source image |
| `perception_fusion` | `perception_fusion_node.py` | Time-synchronises detections + image + LiDAR; projects LiDAR into camera space; routes detections to six typed `perception_msgs` topics |
| `light_classifier` | `light_classifier_node.py` | Crops traffic light bounding boxes and classifies state (red/green/yellow/arrows) using EfficientNet B0 |
| `sign_classifier` | `sign_classifier_node.py` | Crops traffic sign bounding boxes and classifies type (15 US sign categories) using EfficientNet B0 |
| `speed_limit_ocr` | `speed_limit_ocr_node.py` | Reads numeric speed values from speed limit sign crops using EasyOCR |
| `lane_seg` | `lane_seg_node.py` | Produces binary drivable area and lane line segmentation masks using TwinLiteNet |
| `perception_viz` | `perception_viz_node.py` | Overlays all pipeline outputs onto a single annotated image published to `/perception/annotated` |
| `lidar_stamp_relay` | `lidar_stamp_relay.py` | Republishes a LiDAR PointCloud2 topic with a corrected timestamp offset to align with the camera stream |
| `scoring_can` | `scoring_can.py` | Subscribes to all six scoring topics and logs object type, confidence, position, and source to the terminal |

---

## ROS Topics

### Subscriptions (inputs)

| Topic | Type | Used By |
|---|---|---|
| `/synced/camera/image_raw` | `sensor_msgs/Image` | `detector_node` |
| `/synced/lidar/points` | `sensor_msgs/PointCloud2` | `lidar_stamp_relay` |
| `/lidar/points_corrected` | `sensor_msgs/PointCloud2` | `perception_fusion` |
| `/yolov10/detections` | `yolov10_ros/DetectionArray` | `light_classifier`, `sign_classifier`, `perception_fusion`, `perception_viz` |
| `/yolov10/image_raw` | `sensor_msgs/Image` | `light_classifier`, `sign_classifier`, `lane_seg`, `perception_fusion`, `perception_viz` |
| `/perception/traffic_signs` | `yolov10_ros/ClassifiedSignArray` | `speed_limit_ocr`, `perception_fusion` |
| `/perception/traffic_lights` | `yolov10_ros/ClassifiedLightArray` | `perception_fusion` |

### Publications (outputs)

| Topic | Type | Publisher |
|---|---|---|
| `/yolov10/detections` | `yolov10_ros/DetectionArray` | `detector_node` |
| `/yolov10/image_raw` | `sensor_msgs/Image` | `detector_node` |
| `/yolov10/annotated` | `sensor_msgs/Image` | `detector_node` (optional) |
| `/perception/traffic_lights` | `yolov10_ros/ClassifiedLightArray` | `light_classifier` |
| `/perception/traffic_signs` | `yolov10_ros/ClassifiedSignArray` | `sign_classifier` |
| `/perception/speed_limit` | `yolov10_ros/SpeedLimitArray` | `speed_limit_ocr` |
| `/perception/drivable_area` | `sensor_msgs/Image` | `lane_seg` |
| `/perception/lane_lines` | `sensor_msgs/Image` | `lane_seg` |
| `/perception/annotated` | `sensor_msgs/Image` | `perception_viz` |
| `/perception/lidar_overlay` | `sensor_msgs/Image` | `perception_fusion` |
| `/objects_scoring` | `perception_msgs/Objects` | `perception_fusion` |
| `/TrafficSigns_scoring` | `perception_msgs/TrafficSigns` | `perception_fusion` |
| `/TrafficSignalsHead_scoring` | `perception_msgs/TrafficSignalsHead` | `perception_fusion` |
| `/LaneMarkings_scoring` | `perception_msgs/LaneMarkings` | `perception_fusion` |
| `/LimitLines_scoring` | `perception_msgs/LimitLines` | `perception_fusion` |
| `/road_state_scoring` | `perception_msgs/road_state` | `perception_fusion` |

---

## Prerequisites

- **ROS Noetic** (Ubuntu 20.04)
- **Python 3.8+**
- **CUDA** (recommended; CPU mode supported but slow)
- Python packages:
  ```
  torch torchvision
  ultralytics
  opencv-python
  numpy
  easyocr
  ```
- ROS packages:
  ```
  perception_msgs
  sensor_msgs
  cv_bridge
  message_filters
  topic_tools
  ```

---

## Installation

```bash
# 1. Clone into your catkin workspace
cd ~/catkin_ws/src
git clone https://github.com/Christopher-Tetteh-Nenebi/yolov10_ros_ncatutodrive.git yolov10_ros

# 2. Build
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Place your model weights in the `models/` directory:
```
models/
├── yolov10-bdd-vanilla.pt
├── efficientnet_b0_..._sign_cls.pth
├── efficientnet_b0_..._light_cls.pth
└── twinlitenet_best.pth
```

---

## Launch Files

### 1. Detector only

Runs just the YOLOv10 detector. Useful for verifying the model and camera topic work before running the full pipeline.

```bash
roslaunch yolov10_ros detector.launch \
  input_image_topic:=/your/camera/topic \
  device:=0
```

| Argument | Default | Description |
|---|---|---|
| `weights` | `models/yolov10-bdd-vanilla.pt` | Path to YOLOv10 `.pt` weights |
| `confidence_threshold` | `0.5` | Minimum detection confidence |
| `device` | `0` | GPU index or `cpu` |
| `inference_size` | `640` | Model input resolution (px) |
| `input_image_topic` | `/camera_2/...` | Camera topic to subscribe to |
| `view_image` | `true` | Show OpenCV window |

---

### 2. Full pipeline (camera only, no LiDAR)

Runs detector + classifiers + lane segmentation + visualizer. No LiDAR fusion.

```bash
roslaunch yolov10_ros pipeline.launch \
  input_image_topic:=/your/camera/topic \
  device:=0
```

| Argument | Default | Description |
|---|---|---|
| `device` | `0` | GPU index or `cpu` |
| `input_image_topic` | `/camera_2/...` | Camera topic |
| `detector_conf` | `0.5` | Detector confidence threshold |
| `lane_seg` | `true` | Enable lane segmentation |
| `publish_combined` | `true` | Publish combined annotated image |

---

### 3. Pipeline with RealSense D435 camera

Same as above but also launches the Intel RealSense D435 camera driver.

```bash
roslaunch yolov10_ros pipeline_d435.launch device:=0
```

---

### 4. Pipeline with KITTI rosbag

Configures the pipeline for topics published by `kitti2bag`.

```bash
roslaunch yolov10_ros pipeline_kitti.launch \
  bag_image_topic:=/kitti/camera_color_left/image_raw \
  device:=0
```

---

### 5. Full perception stack with LiDAR fusion (recommended)

The main launch file. Runs all nodes including LiDAR-camera fusion and routes detections to `perception_msgs` scoring topics.

```bash
roslaunch yolov10_ros yolov10_perception.launch \
  camera_topic:=/your/camera/topic \
  lidar_topic:=/your/lidar/topic \
  calib_path:=$(rospack find yolov10_ros)/calib_files \
  device:=0 \
  slop:=0.2
```

| Argument | Default | Description |
|---|---|---|
| `weights` | `models/yolov10-bdd-vanilla.pt` | YOLOv10 weights path |
| `calib_path` | `calib_files/` | Directory with calibration files |
| `camera_topic` | `/camera_2/...` | Camera topic |
| `lidar_topic` | `/ouster/points` | LiDAR PointCloud2 topic |
| `confidence_threshold` | `0.5` | Detector confidence |
| `inference_size` | `640` | Model input resolution |
| `device` | `0` | GPU index or `cpu` |
| `half` | `false` | FP16 inference (GPU only) |
| `slop` | `0.1` | ApproxTimeSynchronizer tolerance (seconds) |
| `publish_annotated` | `false` | Publish annotated debug image |

---

## Testing with a ROS Bag

Use this procedure to run the full pipeline against a recorded bag file. Each step runs in a separate terminal.

### Terminal 1 — roscore
```bash
roscore
```

### Terminal 2 — launch the full pipeline
```bash
cd ~/catkin_ws && source devel/setup.bash
roslaunch yolov10_ros yolov10_perception.launch \
  camera_topic:=/synced/camera/image_raw \
  lidar_topic:=/lidar/points_corrected \
  calib_path:=$(rospack find yolov10_ros)/calib_files \
  device:=cpu \
  slop:=0.2
```
> Wait for `[fusion] Calibration loaded` before continuing.

### Terminal 3 — LiDAR timestamp relay
> Required when the bag has a timestamp offset between camera and LiDAR.
> The `synced_dataset.bag` has a known 4.672 s offset.

```bash
source ~/catkin_ws/devel/setup.bash
rosrun yolov10_ros lidar_stamp_relay.py \
  _input_topic:=/synced/lidar/points \
  _output_topic:=/lidar/points_corrected \
  _offset_sec:=-4.672
```

You should see:
```
[lidar_relay] /synced/lidar/points → /lidar/points_corrected  (offset -4.672 s)
```

### Terminal 4 — scoring CAN monitor
```bash
source ~/catkin_ws/devel/setup.bash
rosrun yolov10_ros scoring_can.py
```

### Terminal 5 — play the bag
```bash
rosparam set /use_sim_time true
rosbag play /path/to/your.bag --clock -r 0.5
```
> `-r 0.5` plays at half speed so CPU inference keeps up. Use `-r 1.0` with GPU.

### Terminal 6 — verify
```bash
rostopic hz /objects_scoring       # should show ~7-10 Hz
rostopic echo /objects_scoring     # should show object tracks with positions
```

---

## Scoring CAN Node

The `scoring_can` node subscribes to all six perception scoring topics and logs structured output to the terminal. It is the primary interface for verifying that the pipeline is producing correct perception messages.

```bash
rosrun yolov10_ros scoring_can.py
```

Example output:
```
[scoring] Objects — count=3  t=1500.0 s
  obj id=1  type=car  conf=high  long=33.79 m  lat=-8.99 m  w=274  h=104  src_cam=1  src_lidar=1
  obj id=2  type=car  conf=medium  long=73.95 m  lat=-15.90 m  w=211  h=107  src_cam=1  src_lidar=1
  obj id=3  type=person  conf=high  long=12.40 m  lat=1.20 m  w=60  h=120  src_cam=1  src_lidar=0
[scoring] TrafficSigns — count=1  t=1500.0 s
  sign id=1  type=stop  conf=high  long=20.10 m  lat=0.05 m
[scoring] TrafficSignalsHead — count=1  t=1500.0 s
  signal id=1  state=IllumLtRedBall  conf=high  long=45.20 m  lat=-0.30 m
```

**Field meanings:**

| Field | Description |
|---|---|
| `long` | Forward distance from the vehicle in metres (LiDAR frame x-axis) |
| `lat` | Lateral offset from vehicle centre in metres (negative = left) |
| `src_cam=1` | Detection came from camera |
| `src_lidar=1` | Depth estimated from LiDAR backprojection |
| `src_lidar=0` | No LiDAR points found in bounding box; position unavailable |

---

## Configuration Parameters

All nodes load their default parameters from YAML files in the `config/` directory:

| File | Node | Key Parameters |
|---|---|---|
| `detector_params.yaml` | `detector_node` | `confidence_threshold`, `inference_size`, `half`, `view_image` |
| `light_classifier_params.yaml` | `light_classifier_node` | `confidence_threshold`, `roi_padding`, `target_class` |
| `sign_classifier_params.yaml` | `sign_classifier_node` | `confidence_threshold`, `roi_padding`, `target_class` |
| `speed_limit_ocr_params.yaml` | `speed_limit_ocr_node` | `ocr_confidence_threshold`, `min_crop_height`, `roi_padding` |
| `lane_seg_params.yaml` | `lane_seg_node` | `image_topic`, `drivable_area_topic`, `lane_lines_topic` |

Override any parameter at launch time:
```bash
roslaunch yolov10_ros yolov10_perception.launch \
  confidence_threshold:=0.4 \
  inference_size:=320
```

---

## Calibration Files

The `calib_files/` directory contains camera and LiDAR calibration data required by `perception_fusion_node`:

| File | Contents |
|---|---|
| `calib_cam_to_cam.txt` | Camera intrinsic matrix (3×3) used to project LiDAR into image space |
| `calib_trial_Jan28.txt` | LiDAR-to-camera extrinsic rigid transformation (4×4) |
| `kitti_camera_color_left_right.yaml` | KITTI-format stereo camera calibration |
| `results.txt` | Calibration residuals and reprojection error report |

If using a different sensor setup, replace these files with your own calibration and pass the directory via `calib_path`.

---

## Directory Structure

```
yolov10_ros/
├── calib_files/              Sensor calibration files
├── config/                   YAML parameter files for each node
├── docs/                     Architecture and API documentation
├── launch/
│   ├── detector.launch           Detector only
│   ├── pipeline.launch           Full pipeline (camera only)
│   ├── pipeline_d435.launch      Pipeline + RealSense D435
│   ├── pipeline_kitti.launch     Pipeline for KITTI bags
│   └── yolov10_perception.launch Full stack with LiDAR fusion
├── models/                   Model weights (not tracked in git)
├── msg/                      Custom ROS message definitions
├── src/
│   ├── yolov10_ros/          Python module (image_utils, visualization, kitti_utils, etc.)
│   ├── detector_node.py
│   ├── perception_fusion_node.py
│   ├── light_classifier_node.py
│   ├── sign_classifier_node.py
│   ├── speed_limit_ocr_node.py
│   ├── lane_seg_node.py
│   ├── perception_viz_node.py
│   ├── lidar_stamp_relay.py
│   └── scoring_can.py
└── app.py                    Streamlit web application for offline testing
```

---

## Troubleshooting

**`/objects_scoring` has no messages**
- Check that all nodes are running: `rosnode list`
- Verify all three sync topics are publishing:
  ```bash
  rostopic hz /yolov10/detections /yolov10/image_raw /lidar/points_corrected
  ```
- If camera and LiDAR timestamps differ by several seconds, run `lidar_stamp_relay.py` and increase `slop`

**Slow detections / objects already left the scene when boxes appear**
- Switch from CPU to GPU: `device:=0`
- Reduce inference resolution: `inference_size:=320`
- Enable FP16 (GPU only): `half:=true`

**`src_lidar=0` on all objects / `long=0 lat=0`**
- LiDAR points are not landing inside bounding boxes
- Run `lidar_stamp_relay.py` to correct the timestamp offset between camera and LiDAR
- Check that `calib_files/` are correct for your sensor setup
- Verify LiDAR topic is publishing: `rostopic hz /your/lidar/topic`

**`WARNING: may be using simulated time`**
- Set sim time before playing the bag:
  ```bash
  rosparam set /use_sim_time true
  rosbag play your.bag --clock
  ```
