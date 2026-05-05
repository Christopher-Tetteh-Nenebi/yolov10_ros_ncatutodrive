#!/usr/bin/env bash
# perception_viz.sh — boot the full perception stack against a local rosbag
# and open the LiDAR-overlay viewer in one shot.
#
# Replaces the multi-terminal flow:
#   terminal 1: source devel/setup.bash && roscore
#   terminal 2: rosbag play -l lidar_cam_straight.bag
#   terminal 3: roslaunch yolov10_ros perception_full.launch ...
#   terminal 4: rqt_image_view /perception/lidar_overlay
#
# Usage (from a fresh terminal, working directory does not matter):
#   bash src/yolov10_ros/scripts/perception_viz.sh                  # default bag
#   bash src/yolov10_ros/scripts/perception_viz.sh path/to/x.bag    # custom bag
#
# Optional environment overrides:
#   CAMERA_TOPIC=/your/cam/topic LIDAR_TOPIC=/your/lidar \
#     bash src/yolov10_ros/scripts/perception_viz.sh
#
# Press Ctrl+C in this terminal (or close the rqt window) to stop the whole
# stack cleanly.

set -uo pipefail

# Resolve workspace root: this script lives at <ws>/src/yolov10_ros/scripts/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS_DIR="$( cd "${SCRIPT_DIR}/../../.." && pwd )"

BAG="${1:-${WS_DIR}/lidar_cam_straight.bag}"
CAMERA_TOPIC="${CAMERA_TOPIC:-/camera_1/usb_cam_1/usb_cam_1_node/image_raw}"
LIDAR_TOPIC="${LIDAR_TOPIC:-/ouster/points}"

if [[ ! -f "${BAG}" ]]; then
  echo "ERROR: Bag file not found: ${BAG}" >&2
  echo "Pass a path as the first argument, or place the bag at the workspace root." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
# shellcheck disable=SC1091
source "${WS_DIR}/devel/setup.bash"

declare -a PIDS=()

cleanup() {
  echo
  echo "[perception_viz] Shutting down..."
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
  echo "[perception_viz] Starting roscore..."
  roscore >/tmp/perception_viz_roscore.log 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 30); do
    rostopic list >/dev/null 2>&1 && break
    sleep 0.3
  done
else
  echo "[perception_viz] roscore already running — reusing it."
fi

# 2) rosbag play (looped)
echo "[perception_viz] Playing bag: ${BAG}"
rosbag play -l "${BAG}" >/tmp/perception_viz_bag.log 2>&1 &
PIDS+=($!)

# 3) Wait for camera and LiDAR topics to appear before launching the stack.
for topic in "${CAMERA_TOPIC}" "${LIDAR_TOPIC}"; do
  echo "[perception_viz] Waiting for ${topic}..."
  for _ in $(seq 1 60); do
    rostopic list 2>/dev/null | grep -qx -- "${topic}" && break
    sleep 0.5
  done
done

# 4) Full perception stack with the right camera_topic override.
echo "[perception_viz] Launching perception_full.launch..."
roslaunch yolov10_ros perception_full.launch \
  camera_topic:="${CAMERA_TOPIC}" \
  lidar_topic:="${LIDAR_TOPIC}" \
  >/tmp/perception_viz_launch.log 2>&1 &
PIDS+=($!)

# 5) Wait for the overlay topic to be advertised (≈ models loaded).
echo "[perception_viz] Loading models (~20-30s)..."
for _ in $(seq 1 60); do
  rostopic list 2>/dev/null | grep -q "/perception/lidar_overlay" && break
  sleep 1
done

# 6) Viewer in the foreground — closes the loop on Ctrl+C / window close.
echo "[perception_viz] Opening rqt_image_view /perception/lidar_overlay"
echo "[perception_viz] Press Ctrl+C here, or close the rqt window, to stop."
rqt_image_view /perception/lidar_overlay
