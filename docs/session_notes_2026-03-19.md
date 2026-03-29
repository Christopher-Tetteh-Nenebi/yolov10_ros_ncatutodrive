# Session Notes — 2026-03-19

## Overview
Bring-up and debugging of the full YOLOv10 perception pipeline on a live ROS Noetic system, from a cold start to a streaming lidar-camera fusion overlay.

---

## 1. Package Overview

**`yolov10_ros`** is a ROS Noetic perception package built around the YOLOv10 object detector trained on BDD100K (10 classes: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign). It consists of:

| Node | Role |
|---|---|
| `detector_node.py` | Runs YOLOv10 inference on camera images, publishes `DetectionArray` |
| `perception_fusion_node.py` | Syncs detections + image + LiDAR; projects LiDAR for depth; routes to perception message topics |
| `sign_classifier_node.py` | Sub-classifies traffic sign crops (stop, yield, speed limit, etc.) |
| `light_classifier_node.py` | Sub-classifies traffic light crops (red, yellow, green) |
| `perception_viz_node.py` | Draws bounding boxes, lane lines, drivable area, and sign/light overlays |

Published perception topics:

| Topic | Message Type | Content |
|---|---|---|
| `/yolov10/detections` | `DetectionArray` | Raw YOLO detections |
| `/objects_scoring` | `ObjectsScoringList` | Fused object tracks (car, truck, etc.) |
| `/TrafficSigns_scoring` | `TrafficSignsScoringList` | Traffic sign tracks (stub — see §7) |
| `/perception/annotated` | `sensor_msgs/Image` | Annotated camera image from viz node |
| `/perception/lidar_overlay` | `sensor_msgs/Image` | LiDAR projected onto camera with depth labels |

---

## 2. Calibration Files Created

Created `calib_files/` directory and populated two files with realistic random values:

### `calib_cam_to_cam.txt`
Standard KITTI-format camera intrinsic calibration. Line index 2 is parsed as the 3×3 intrinsic matrix `K`:
```
K_00: 7.188560e+02 0.000000e+00 6.071928e+02 ...
```

### `calib_trial_Jan28.txt`
KITTI-format LiDAR-to-camera extrinsic calibration. Contains rotation matrix `R` (3×3) and translation vector `T` (3×1) for the velodyne→camera rigid transform.

Both files are read by `kitti_utils.get_rigid_transformation()` to build the 4×4 `T_velo_cam` homogeneous transform used for LiDAR point projection.

---

## 3. Launch File: `yolov10_perception.launch`

### Created from scratch with:
- `calib_path` param pointing to `calib_files/` directory
- `camera_topic` set to `/camera_2/usb_cam_2/usb_cam_2_node/image_raw`
- `weights` pointing to `models/yolov10-bdd-vanilla.pt`

### Nodes added incrementally during session:
1. `yolov10_detector`
2. `perception_fusion`
3. `perception_viz_node` (added after user confirmed detections working)
4. `sign_classifier_node` (added after stop sign not visible in overlay)

### Still not added (oversight):
- `light_classifier_node` — present in `pipeline.launch`, never ported over

---

## 4. Fix: `ModuleNotFoundError: No module named 'yolov10_ros.msg'`

**Root cause:** The launch file used `<env name="PYTHONPATH">` which completely replaced the environment variable, stripping the catkin devel path where `yolov10_ros.msg` (generated message classes) lives.

**Fix:** Removed PYTHONPATH manipulation entirely. Instead:
- Copied `kitti_utils.py` from `yolov5_ros/src/yolov5/` → `src/yolov10_ros/kitti_utils.py`
- Copied `bdd100k_class_map.py` → `src/yolov10_ros/bdd100k_class_map.py`
- Updated imports in `perception_fusion_node.py` to:
  ```python
  from yolov10_ros.kitti_utils import get_rigid_transformation, project_velobin2uvz
  from yolov10_ros.bdd100k_class_map import get_object_type, categorize_confidence
  ```

---

## 5. Detector Confirmed Working

```
rostopic echo /yolov10/detections
```
Sample output showed high-confidence detections updating per frame:
- `traffic sign` — 0.94 confidence
- `truck` — 0.76 confidence
- `car` — 0.69 confidence

---

## 6. Fusion Node Confirmed Working

```
rostopic echo /objects_scoring
```

| Field | Status |
|---|---|
| `Current_Object_Count` | Correct |
| `ObjObjectType` | Correctly mapped (car/truck → type 1) |
| `Width` / `Height` | Populated from bounding box dimensions |
| `Confidence` | 1–2 (medium, from categorize_confidence()) |
| `Object_Source_Camera: 1` | Correctly flagged |
| `LongPos` / `LatPos` | `0.0` — no LiDAR in sync yet (graceful fallback) |

---

## 7. What is a Stub?

A stub is a placeholder implementation that has the correct structure but does no real work.

**Example — fully implemented (`_build_objects`):**
```python
track.ObjObjectType  = get_object_type(name)
track.Confidence     = categorize_confidence(conf)
track.Width          = xmax - xmin
```

**Example — stub (`_build_trafficsigns`):**
```python
for _ in items:
    msg.trafficSign_track.append(TrafficSign_Track())  # all zeros
```

The `_` means "I'm not using this value." The correct number of entries are created, but every field stays at its zero-default. This is why `/TrafficSigns_scoring` showed `Current_Sign_Count: 0` and all zeros — the detection existed but was ignored.

---

## 8. Stop Sign Not Visible in Viz Overlay

**Root cause:** `perception_viz_node.py` intentionally skips raw `"traffic sign"` and `"traffic light"` boxes from the YOLO detector — it waits for the downstream sub-classifiers to provide a specific label (stop, yield, red, green, etc.). The `sign_classifier_node` was not running.

**Fix:** Added `sign_classifier_node` to `yolov10_perception.launch`.

---

## 9. LiDAR-Camera Fusion Overlay

### Problem
The fusion node already projected LiDAR points and drew depth text on an image (`draw=True` in `_get_uvz_centers`), but that local `image` variable was never published.

### Fix — Added `/perception/lidar_overlay` publisher
A separate image subscriber (`_cb_image`) was added to `perception_fusion_node.py`:
- Caches the latest flat LiDAR pointcloud in `_cached_flat_pcl` via `_cb_pcl`
- On every camera frame, projects cached LiDAR and draws coloured depth dots using `draw_velo_on_image`
- Draws bounding boxes with inline depth labels (e.g. `car 0.85 | 12.34m`)
- Publishes to `/perception/lidar_overlay` at camera frame rate (~12 Hz)

### Depth label fix
Initial approach cached `(xmin, ymin, xmax, ymax) → depth` from the 3-way sync callback and looked it up in `_cb_image`. This failed because bounding box coordinates change frame-to-frame between the two callbacks.

**Final fix:** Depth is computed directly inside `_cb_image` from the projected `velo_uvz` using the same L2-nearest-point logic as `_get_uvz_centers`:
```python
delta   = np.abs(np.array((v_arr, u_arr)) - np.array([[cy, cx]]).T)
min_loc = np.argmin(np.linalg.norm(delta, axis=0))
depth   = z_arr[min_loc]
```
This is independent of the 3-way `ApproximateTimeSynchronizer`.

---

## 10. Open Items

| Item | Status |
|---|---|
| `_build_trafficsigns()` stub — fill in real field mapping | Pending |
| `_build_trafficsignals()` stub | Pending |
| `_build_lanemarkings()` stub | Pending |
| `light_classifier_node` missing from `yolov10_perception.launch` | Pending |
| Real calibration files (replace random values with surveyed extrinsics) | Pending |
| LiDAR depth in `/objects_scoring` `LongPos`/`LatPos` fields | Pending (requires 3-way sync to fire consistently) |
