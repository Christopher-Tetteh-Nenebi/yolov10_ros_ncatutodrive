#!/usr/bin/env bash
# perception_viz_live.sh — boot the full perception stack against the LIVE
# middle camera + Ouster LiDAR and open the LiDAR-overlay viewer.
#
# Replaces the multi-terminal flow:
#   terminal 1: source devel/setup.bash && roscore
#   terminal 2: roslaunch lidar_cams_setup lidar_cam_middle_bringup.launch
#   terminal 3: roslaunch yolov10_ros perception_full.launch
#   terminal 4: rqt_image_view /perception/lidar_overlay
#
# Camera 2 (middle) publishes /camera_2/usb_cam_2/usb_cam_2_node/image_raw,
# which is also the default camera_topic in perception_full.launch — no
# topic override is required.
#
# Usage (from a fresh terminal, working directory does not matter):
#   bash src/yolov10_ros/scripts/perception_viz_live.sh
#
# Optional environment overrides:
#   FRAMERATE=15 \
#   SENSOR_HOSTNAME=os-122345678901.local \
#   bash src/yolov10_ros/scripts/perception_viz_live.sh
#
# Press Ctrl+C in this terminal (or close the rqt window) to stop the whole
# stack cleanly.

set -uo pipefail

# Resolve workspace root: this script lives at <ws>/src/yolov10_ros/scripts/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS_DIR="$( cd "${SCRIPT_DIR}/../../.." && pwd )"

CAMERA_TOPIC="${CAMERA_TOPIC:-/camera_2/usb_cam_2/usb_cam_2_node/image_raw}"
LIDAR_TOPIC="${LIDAR_TOPIC:-/ouster/points}"
FRAMERATE="${FRAMERATE:-30}"
SENSOR_HOSTNAME="${SENSOR_HOSTNAME:-os-122345678901.local}"

# Sanity check: the middle camera's /dev/video device should exist.
if [[ ! -e /dev/video2 ]]; then
  echo "WARNING: /dev/video2 not present — the middle camera may be unplugged." >&2
fi

# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
# shellcheck disable=SC1091
source "${WS_DIR}/devel/setup.bash"

declare -a PIDS=()

cleanup() {
  echo
  echo "[perception_viz_live] Shutting down..."
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill -INT  "${pid}" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill -KILL "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

# 1) roscore — only if not already up
if ! rostopic list >/dev/null 2>&1; then
  echo "[perception_viz_live] Starting roscore..."
  roscore >/tmp/perception_viz_live_roscore.log 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 30); do
    rostopic list >/dev/null 2>&1 && break
    sleep 0.3
  done
else
  echo "[perception_viz_live] roscore already running — reusing it."
fi

# 2) Live sensor bringup: middle camera + Ouster LiDAR.
echo "[perception_viz_live] Starting middle camera + Ouster LiDAR..."
roslaunch lidar_cams_setup lidar_cam_middle_bringup.launch \
  framerate:="${FRAMERATE}" \
  sensor_hostname:="${SENSOR_HOSTNAME}" \
  >/tmp/perception_viz_live_sensors.log 2>&1 &
PIDS+=($!)

# 3) Wait for camera and LiDAR topics to start publishing.
for topic in "${CAMERA_TOPIC}" "${LIDAR_TOPIC}"; do
  echo "[perception_viz_live] Waiting for ${topic}..."
  for _ in $(seq 1 60); do
    rostopic list 2>/dev/null | grep -qx -- "${topic}" && break
    sleep 0.5
  done
done

# 4) Full perception stack with default topics (already match camera_2 + ouster).
echo "[perception_viz_live] Launching perception_full.launch..."
roslaunch yolov10_ros perception_full.launch \
  camera_topic:="${CAMERA_TOPIC}" \
  lidar_topic:="${LIDAR_TOPIC}" \
  >/tmp/perception_viz_live_launch.log 2>&1 &
PIDS+=($!)

# 5) Wait for the overlay topic to be advertised (≈ models loaded).
echo "[perception_viz_live] Loading models (~20-30s)..."
for _ in $(seq 1 60); do
  rostopic list 2>/dev/null | grep -q "/perception/lidar_overlay" && break
  sleep 1
done

# 6) Viewer in the foreground.
echo "[perception_viz_live] Opening rqt_image_view /perception/lidar_overlay"
echo "[perception_viz_live] Press Ctrl+C here, or close the rqt window, to stop."
rqt_image_view /perception/lidar_overlay
