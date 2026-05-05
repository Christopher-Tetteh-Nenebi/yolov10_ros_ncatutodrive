#!/usr/bin/env python3
from __future__ import annotations
"""
Perception Fusion Node
======================
Bridges the thin YOLOv10 detector to the rest of the perception stack.

Responsibilities
----------------
1. Time-synchronise YOLOv10 detections + source image with the LiDAR stream.
2. Project LiDAR points into camera space and fuse depth into each bounding box.
3. Route detections to the six typed perception topics consumed by downstream
   modules (objects, traffic signs, signal heads, lane markings, limit lines,
   road state).

This node deliberately owns *no* inference code.  It depends on
``detector_node.py`` for detections and the decoded source image.

Subscriptions
-------------
  ~input_detections_topic  (yolov10_ros/DetectionArray)
      Bounding-box detections from the YOLOv10 detector node.
      Default: /yolov10/detections

  ~input_image_topic       (sensor_msgs/Image)
      Decoded source frame republished by the detector node.
      Default: /yolov10/image_raw

  ~input_pcl_topic         (sensor_msgs/PointCloud2)
      LiDAR point cloud (Ouster format: x, y, z, range in mm).

Publications
------------
  /objects_scoring              (perception_msgs/Objects)
  /TrafficSigns_scoring         (perception_msgs/TrafficSigns)
  /TrafficSignalsHead_scoring   (perception_msgs/TrafficSignalsHead)
  /LaneMarkings_scoring         (perception_msgs/LaneMarkings)
  /LimitLines_scoring           (perception_msgs/LimitLines)
  /road_state_scoring           (perception_msgs/road_state)

ROS Parameters
--------------
  ~input_detections_topic  (str)   Default: /yolov10/detections
  ~input_image_topic       (str)   Default: /yolov10/image_raw
  ~input_pcl_topic         (str)   LiDAR topic (required).
  ~calib_path              (str)   Directory containing calib_cam_to_cam.txt
                                   and calib_trial_Jan28.txt.
  ~slop                    (float) ApproximateTimeSynchronizer tolerance (s).
                                   Default: 0.1

Notes
-----
- LiDAR depth fusion is wrapped in a try/except.  On failure the detections
  are still routed to perception topics with (x, y, z) = (0, 0, 0) so
  downstream modules keep receiving data.
- The Ouster ``range`` field is in millimetres; it is converted to metres
  internally before projection.
- Calibration files are read once at startup, not on every callback.
"""

import os
import logging

import rospy
import cv2
import numpy as np

from datetime import datetime
from collections import defaultdict
from rostopic import get_topic_type

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer

from yolov10_ros.kitti_utils import get_rigid_transformation, project_velobin2uvz, draw_velo_on_image

from yolov10_ros.msg import DetectionArray
from yolov10_ros.image_utils import imgmsg_to_cv2, cv2_to_imgmsg

from perception_msgs.msg import (
    Objects, Objects_Track,
    TrafficSigns, TrafficSign_Track,
    TrafficSignalsHead, TrafficSignalsHead_Track,
    LaneMarkings, LaneMarking_Track,
    LimitLines, LimitLine_Track,
    road_state,
)

from yolov10_ros.bdd100k_class_map import (
    get_topic_key,
    get_object_type,
    categorize_confidence,
)

logging.basicConfig(level=logging.ERROR)

# ── Traffic sign type → Sign_Type integer ─────────────────────────────────────
# Matches the 15-class mapping from sign_classifier_node._DEFAULT_CLASSES.
# Downstream modules index Sign_Type as an enum; 0 = unknown.
_SIGN_TYPE_MAP: dict[str, int] = {
    "stop":                 1,
    "yield":                2,
    "speed_limit":          3,
    "no_left_turn":         4,
    "no_right_turn":        5,
    "no_u_turn":            6,
    "no_straight":          7,
    "go_straight_only":     8,
    "turn_left_only":       9,
    "turn_right_only":      10,
    "pedestrian_crossing":  11,
    "railroad_crossing":    12,
    "roadwork":             13,
    "detour":               14,
    "do_not_enter":         15,
}

# ── Light state → TrafficSignalsHead_Track illumination field name ─────────────
# Each entry maps a classifier output string to the field that should be set to 1.
_LIGHT_ILLUM_MAP: dict[str, str] = {
    "red_light":    "IllumLtRedBall",
    "green_light":  "IllumLtGreenBall",
    "yellow_light": "IllumLtYellowBall",
    "red_left":     "IllumLtRedLeftArrow",
    "green_left":   "IllumLtGreenLeftArrow",
    "yellow_left":  "IllumLtYellowLeftArrow",
}


class PerceptionFusionNode:
    """
    Fuses YOLOv10 detections with LiDAR depth and publishes perception messages.

    Architecture note
    -----------------
    The detector node publishes detections and a decoded source image on every
    camera frame.  This node synchronises those two topics with the LiDAR
    PointCloud2 stream using ApproximateTimeSynchronizer.  The sync tolerance
    (``slop``) is configurable; 0.1 s is a safe default for 10 Hz LiDAR.
    """

    def __init__(self):
        # ── Calibration (loaded once at startup) ──────────────────────────────
        calib_path = rospy.get_param("~calib_path")
        self.T_velo_cam, self.T_cam_velo = self._load_calibration(calib_path)
        rospy.loginfo("[fusion] Calibration loaded from %s", calib_path)

        # ── Subscribers via message_filters ───────────────────────────────────
        det_topic = rospy.get_param("~input_detections_topic", "/yolov10/detections")
        img_topic = rospy.get_param("~input_image_topic",      "/yolov10/image_raw")
        pcl_param = rospy.get_param("~input_pcl_topic")

        # Resolve the actual topic name and type for the LiDAR topic.
        pcl_type, pcl_topic, _ = get_topic_type(pcl_param, blocking=True)
        if pcl_type != "sensor_msgs/PointCloud2":
            rospy.logwarn(
                "[fusion] Expected PointCloud2 on %s, got %s — proceeding anyway",
                pcl_topic, pcl_type,
            )

        # ── 3-way ATS: detections + image + LiDAR → perception topics ────────
        self.det_sub = Subscriber(det_topic, DetectionArray)
        self.img_sub = Subscriber(img_topic, Image)
        self.pcl_sub = Subscriber(pcl_topic, PointCloud2)

        slop = rospy.get_param("~slop", 0.1)
        self.ats = ApproximateTimeSynchronizer(
            [self.det_sub, self.img_sub, self.pcl_sub],
            queue_size=10,
            slop=slop,
        )
        self.ats.registerCallback(self.callback)

        # ── Overlay subscribers ────────────────────────────────────────────────
        # det + image share the same header timestamp (same detector callback),
        # so TimeSynchronizer (exact match) guarantees frame-perfect alignment.
        # LiDAR uses an independent subscriber — its clock may differ from the
        # camera clock, so we avoid timestamp-based sync for it.
        self._latest_pcl_msg = None
        from message_filters import TimeSynchronizer
        self.det_sub2 = Subscriber(det_topic, DetectionArray)
        self.img_sub2 = Subscriber(img_topic, Image)
        self.ts_det_img = TimeSynchronizer(
            [self.det_sub2, self.img_sub2], queue_size=10
        )
        self.ts_det_img.registerCallback(self._cb_overlay)
        rospy.Subscriber(pcl_topic, PointCloud2, self._cb_pcl, queue_size=1)

        rospy.loginfo(
            "[fusion] Synchronising:\n  detections : %s\n  image      : %s\n  lidar      : %s  (slop=%.2fs)",
            det_topic, img_topic, pcl_topic, slop,
        )


        # ── Classifier result subscribers (independent — no sync required) ────
        # sign_classifier and light_classifier publish at camera frame rate on
        # their own topics.  We cache the latest array and match detections by
        # bounding-box centre proximity inside the message builders.
        from yolov10_ros.msg import ClassifiedSignArray, ClassifiedLightArray, SpeedLimitArray
        self._latest_signs  = []   # list[ClassifiedSign]
        self._latest_lights = []   # list[ClassifiedLight]
        self._latest_speeds = []   # list[SpeedLimit]
        # Wall-clock stamps for TTL-based persistence in the overlay.  When a
        # classifier skips a frame (e.g. softmax below threshold) we keep the
        # last result for OVERLAY_PERSIST_SEC so the box doesn't flicker back
        # to the default color for a single missed frame.
        self._signs_stamp  = rospy.Time(0)
        self._lights_stamp = rospy.Time(0)
        self._speeds_stamp = rospy.Time(0)
        self._da_mask = None       # drivable-area mask (mono8 numpy array)
        self._ll_mask = None       # lane-lines  mask (mono8 numpy array)
        rospy.Subscriber("/perception/traffic_signs",  ClassifiedSignArray,  self._cb_signs,  queue_size=1)
        rospy.Subscriber("/perception/traffic_lights", ClassifiedLightArray, self._cb_lights, queue_size=1)
        rospy.Subscriber("/perception/speed_limit",    SpeedLimitArray,      self._cb_speeds, queue_size=1)
        rospy.Subscriber("/perception/drivable_area",  Image,                self._cb_da,     queue_size=1)
        rospy.Subscriber("/perception/lane_lines",     Image,                self._cb_ll,     queue_size=1)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_objects     = rospy.Publisher("/objects_scoring",             Objects,           queue_size=10)
        self.pub_trafficsigns= rospy.Publisher("/TrafficSigns_scoring",        TrafficSigns,      queue_size=10)
        self.pub_sighead     = rospy.Publisher("/TrafficSignalsHead_scoring",   TrafficSignalsHead,queue_size=10)
        self.pub_lane        = rospy.Publisher("/LaneMarkings_scoring",         LaneMarkings,      queue_size=10)
        self.pub_limit       = rospy.Publisher("/LimitLines_scoring",           LimitLines,        queue_size=10)
        self.pub_road        = rospy.Publisher("/road_state_scoring",           road_state,        queue_size=10)
        self.pub_lidar_img   = rospy.Publisher("/perception/lidar_overlay",    Image,             queue_size=1)

    # ── Overlay callbacks ─────────────────────────────────────────────────────

    def _cb_pcl(self, pcl_msg: PointCloud2):
        self._latest_pcl_msg = pcl_msg

    def _cb_overlay(self, det_msg: DetectionArray, img_msg: Image):
        """Fires when a perfectly matched det+image pair arrives (same timestamp).
        Projects the latest LiDAR scan onto that image and publishes the overlay."""
        if self.pub_lidar_img.get_num_connections() == 0:
            return
        if self._latest_pcl_msg is None:
            return
        image        = imgmsg_to_cv2(img_msg)
        img_h, img_w = image.shape[:2]
        velo_uvz     = self._project_pcl(self._latest_pcl_msg, img_h, img_w)
        overlay      = self._draw_overlay(image, velo_uvz, det_msg.detections)
        self.pub_lidar_img.publish(cv2_to_imgmsg(overlay, 'bgr8'))

    def _project_pcl(self, pcl_msg: PointCloud2, img_h: int, img_w: int):
        """Fast numpy PointCloud2 → camera (u, v, z) projection.

        Reads the raw byte buffer directly (no Python iterator) so all
        ~130k Ouster points are processed in <10 ms.

        Returns:
            velo_uvz : ndarray (3, M) of [u, v, z] for in-frame points,
                       or None on failure.
        """
        try:
            fields = {f.name: f.offset for f in pcl_msg.fields}
            ps     = pcl_msg.point_step
            n_pts  = pcl_msg.width * pcl_msg.height
            raw    = np.frombuffer(pcl_msg.data, dtype=np.uint8).reshape(n_pts, ps)
            x = raw[:, fields['x']:fields['x'] + 4].copy().view(np.float32).ravel()
            y = raw[:, fields['y']:fields['y'] + 4].copy().view(np.float32).ravel()
            z = raw[:, fields['z']:fields['z'] + 4].copy().view(np.float32).ravel()
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[fusion] PCL read failed: %s", exc)
            return None

        # Euclidean distance from sensor — filters vehicle body returns (~0–1.5 m)
        # and keeps only points more than MIN_RANGE metres away.
        MIN_RANGE = 1.5   # metres — lowered from 3.0 to capture nearby objects
        dist  = np.sqrt(x*x + y*y + z*z)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0.0) & (dist > MIN_RANGE)
        pts   = np.column_stack([x[valid], y[valid], z[valid]])
        if len(pts) == 0:
            return None

        xyzw = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)]).T
        uvz  = self.T_velo_cam.astype(np.float32) @ xyzw   # (3, N)

        front    = uvz[2] > 0
        uvz      = uvz[:, front]
        uvz[:2] /= uvz[2:3]   # perspective divide

        in_frame = (
            (uvz[0] >= 0) & (uvz[0] < img_w) &
            (uvz[1] >= 0) & (uvz[1] < img_h)
        )
        return uvz[:, in_frame]   # (3, M): u, v, z

    def _draw_overlay(self, image, velo_uvz, detections):
        """Draw LiDAR dots + bounding boxes with depth labels onto image.

        Boxes are colored by classifier output when available:
          - traffic light  → red / yellow / green by predicted state
          - traffic sign   → sky-blue with sign_type label,
                             red with "SPEED N mph" if OCR value is available
          - other classes  → green with class name
        """
        overlay = image.copy()

        # Drivable-area mask (semi-transparent blue) — drawn behind lidar dots.
        if self._da_mask is not None and self._da_mask.shape == overlay.shape[:2]:
            tint = overlay.copy()
            tint[self._da_mask > 127] = (255, 0, 0)
            overlay = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)

        # Lane-lines mask (semi-transparent green) — also behind lidar dots.
        if self._ll_mask is not None and self._ll_mask.shape == overlay.shape[:2]:
            tint = overlay.copy()
            tint[self._ll_mask > 127] = (0, 255, 0)
            overlay = cv2.addWeighted(overlay, 0.70, tint, 0.30, 0)

        if velo_uvz is not None and velo_uvz.shape[1] > 0:
            draw_velo_on_image(velo_uvz, overlay)

        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = 0.5
        thickness   = 1
        color_default = (  0, 255,   0)   # green — vehicles / persons / etc.
        color_depth   = (  0,   0, 255)   # red   — depth value (always)
        color_sign    = (255, 180,   0)   # sky-blue
        color_speed   = (  0,   0, 255)   # red — speed limit OCR
        light_colors = {
            'green_light':  ( 30, 200,  30),
            'green_left':   ( 30, 200,  30),
            'red_light':    (  0,   0, 220),
            'red_left':     (  0,   0, 220),
            'yellow_light': (  0, 200, 220),
            'yellow_left':  (  0, 200, 220),
        }

        # TTL-gate the classifier caches.  If the classifier hasn't published a
        # fresh frame within OVERLAY_PERSIST_SEC, treat the cache as empty so a
        # genuinely-gone object stops being colored as red/green/yellow forever.
        OVERLAY_PERSIST_SEC = 0.9
        now = rospy.Time.now()
        ttl = rospy.Duration.from_sec(OVERLAY_PERSIST_SEC)
        light_list = self._latest_lights if (now - self._lights_stamp) < ttl else []
        sign_list  = self._latest_signs  if (now - self._signs_stamp)  < ttl else []
        speed_list = self._latest_speeds if (now - self._speeds_stamp) < ttl else []

        def _nearest(items, det):
            """Return the cached item whose box centre is closest to det's,
            within MATCH_RADIUS pixels.  Tolerates ~30 px of frame-to-frame
            bbox jitter so a single classifier miss doesn't cause flicker."""
            MATCH_RADIUS = 30.0
            if not items:
                return None
            det_cx = (det.xmin + det.xmax) / 2.0
            det_cy = (det.ymin + det.ymax) / 2.0
            best, best_d2 = None, MATCH_RADIUS * MATCH_RADIUS
            for it in items:
                cx = (it.xmin + it.xmax) / 2.0
                cy = (it.ymin + it.ymax) / 2.0
                d2 = (cx - det_cx) ** 2 + (cy - det_cy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = it
            return best

        u_arr = velo_uvz[0] if velo_uvz is not None else None
        v_arr = velo_uvz[1] if velo_uvz is not None else None
        z_arr = velo_uvz[2] if velo_uvz is not None else None

        for d in detections:
            depth = self._estimate_depth(u_arr, v_arr, z_arr, d) \
                    if u_arr is not None else None

            box_color = color_default
            label = '{} {:.2f}'.format(d.class_name, d.confidence)
            box_thickness = 2

            if d.class_name == 'traffic light':
                lt = _nearest(light_list, d)
                if lt is not None:
                    box_color = light_colors.get(lt.state, color_default)
                    label = '{} | {} {:.2f}'.format(
                        d.class_name, lt.state, lt.state_confidence)
                    box_thickness = 3
            elif d.class_name == 'traffic sign':
                sg = _nearest(sign_list, d)
                if sg is not None:
                    if sg.sign_type == 'speed_limit':
                        sl = _nearest(speed_list, d)
                        speed_txt = 'SPEED {} mph'.format(sl.speed) if sl is not None \
                                    else 'SPEED ? mph'
                        label = '{} | {}'.format(d.class_name, speed_txt)
                        box_color = color_speed
                        box_thickness = 4
                    else:
                        label = '{} | {} {:.2f}'.format(
                            d.class_name, sg.sign_type, sg.type_confidence)
                        box_color = color_sign
                        box_thickness = 3

            cv2.rectangle(overlay, (d.xmin, d.ymin), (d.xmax, d.ymax),
                          box_color, box_thickness)

            txt_x, txt_y = d.xmin, max(d.ymin - 5, 10)
            if depth is not None and depth > 0:
                cv2.putText(overlay, label + ' |', (txt_x, txt_y),
                            font, font_scale, box_color, thickness, cv2.LINE_AA)
                (w_det, _), _ = cv2.getTextSize(
                    label + ' |', font, font_scale, thickness)
                cv2.putText(overlay, ' {:.2f}m'.format(depth),
                            (txt_x + w_det, txt_y),
                            font, font_scale, color_depth, thickness, cv2.LINE_AA)
            else:
                cv2.putText(overlay, label, (txt_x, txt_y),
                            font, font_scale, box_color, thickness, cv2.LINE_AA)

        return overlay

    # ── Calibration helpers ───────────────────────────────────────────────────

    def _backproject_to_lidar(self, u: float, v: float, z_cam: float):
        """Back-project an image point (u, v) with camera-frame depth z_cam
        to the LiDAR (vehicle) reference frame.

        Steps:
          1. Undo perspective division: X_cam = (u-cx)*z/fx, Y_cam = (v-cy)*z/fy
          2. Apply T_ref0_velo (inv of LiDAR→cam extrinsic) to get LiDAR xyz.

        Returns:
            (x_lidar, y_lidar) where x is forward (LongPos) and y is lateral
            (LatPos), both in metres in the LiDAR/vehicle frame.
        """
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        X_cam  = (u - cx) * z_cam / fx
        Y_cam  = (v - cy) * z_cam / fy
        pt_cam = np.array([X_cam, Y_cam, z_cam, 1.0])
        pt_lidar = self.T_ref0_velo @ pt_cam
        return float(pt_lidar[0]), float(pt_lidar[1])

    def _estimate_depth(self, u_arr, v_arr, z_arr, d) -> float:
        """Robust LiDAR depth estimate for a single bounding box.

        Three search regions are tried in order, each with its own minimum
        point threshold.  Stricter thresholds are applied to outer regions
        because points outside the box are less likely to belong to the object.

        Regions and thresholds
        ----------------------
        inner box (central 60%)  : MIN_INNER  = 2 points
        full bounding box        : MIN_FULL   = 3 points
        expanded box (+50% each side): MIN_EXPANDED = 5 points AND std < 2 m

        If no region meets its threshold, returns None so the caller knows
        depth is unavailable rather than reporting a bad value.

        Depth selection
        ---------------
        Points beyond MAX_OBJECT_DEPTH (80 m) are discarded as background.
        The nearest 0.5 m bin with enough hits is returned (median of that
        bin), so road-surface / wall contamination at larger depth loses.
        """
        MAX_OBJECT_DEPTH  = 80.0   # metres — discard background beyond this
        MIN_INNER         = 2      # minimum points required in inner box
        MIN_FULL          = 3      # minimum points required in full box
        MIN_EXPANDED      = 5      # minimum points required in expanded box
        EXPANDED_MAX_STD  = 2.0    # expanded points must agree within this std (m)

        w = d.xmax - d.xmin
        h = d.ymax - d.ymin

        # ── Region 1: inner box (central 60%) ────────────────────────────────
        shrink = 0.20
        ix1, ix2 = d.xmin + shrink * w, d.xmax - shrink * w
        iy1, iy2 = d.ymin + shrink * h, d.ymax - shrink * h
        inner_mask = (u_arr >= ix1) & (u_arr <= ix2) & (v_arr >= iy1) & (v_arr <= iy2)
        z_inner = z_arr[inner_mask]
        z_inner = z_inner[z_inner <= MAX_OBJECT_DEPTH]

        if len(z_inner) >= MIN_INNER:
            return self._cluster_depth(z_inner)

        # ── Region 2: full bounding box ───────────────────────────────────────
        full_mask = (
            (u_arr >= d.xmin) & (u_arr <= d.xmax) &
            (v_arr >= d.ymin) & (v_arr <= d.ymax)
        )
        z_full = z_arr[full_mask]
        z_full = z_full[z_full <= MAX_OBJECT_DEPTH]

        if len(z_full) >= MIN_FULL:
            return self._cluster_depth(z_full)

        # ── Region 3: expanded box (+50%) ────────────────────────────────────
        expand = 0.5
        ex1 = d.xmin - expand * w
        ex2 = d.xmax + expand * w
        ey1 = d.ymin - expand * h
        ey2 = d.ymax + expand * h
        exp_mask = (u_arr >= ex1) & (u_arr <= ex2) & (v_arr >= ey1) & (v_arr <= ey2)
        z_exp = z_arr[exp_mask]
        z_exp = z_exp[z_exp <= MAX_OBJECT_DEPTH]

        if len(z_exp) >= MIN_EXPANDED and float(np.std(z_exp)) <= EXPANDED_MAX_STD:
            return self._cluster_depth(z_exp)

        # Not enough reliable points in any region.
        return None

    def _cluster_depth(self, z_points: np.ndarray) -> float:
        """Return the depth of the nearest dense cluster in z_points.

        Walks 0.5 m histogram bins from nearest to farthest and returns
        the median of the first bin that contains at least 2 points.
        Falls back to the 10th percentile when all points fall in one bin
        (very close object or extremely sparse scan).
        """
        if len(z_points) == 1:
            return float(z_points[0])

        bin_size = 0.5
        z_min = float(np.min(z_points))
        z_max = float(np.max(z_points))

        if z_max - z_min > bin_size:
            bins = np.arange(z_min, z_max + bin_size, bin_size)
            counts, edges = np.histogram(z_points, bins=bins)
            for i in range(len(counts)):
                if counts[i] >= 2:
                    lo, hi = edges[i], edges[i + 1]
                    cluster = z_points[(z_points >= lo) & (z_points < hi)]
                    return float(np.median(cluster))

        # All points in one bin — use a low percentile as foreground proxy.
        return float(np.percentile(z_points, 10))

    def _load_calibration(self, calib_path: str):
        """
        Load camera and LiDAR calibration files and compute projection matrices.

        Returns:
            T_velo_cam : (3, 4) ndarray — LiDAR → camera projection
            T_cam_velo : (4, 4) ndarray — camera → LiDAR homogeneous transform
        """
        cam_to_cam_file  = os.path.join(calib_path, "calib_cam_to_cam.txt")
        velo_to_cam_file = os.path.join(calib_path, "calib_trial_Jan28.txt")

        with open(cam_to_cam_file, "r") as f:
            calib = f.readlines()

        # Intrinsic matrix from line 2 of cam_to_cam (3×3, no distortion column).
        p_cam = np.array(
            [float(x) for x in calib[2].strip().split(" ")[1:]]
        ).reshape((3, 3))
        # Extend to 3×4 by appending a zero translation column.
        p_cam = np.hstack((p_cam, np.zeros((3, 1))))

        T_velo_ref0 = get_rigid_transformation(velo_to_cam_file)
        T_velo_cam  = p_cam @ T_velo_ref0   # LiDAR → image plane (3×4)

        # Homogeneous inverse: image → LiDAR (4×4).
        T_cam_velo = np.linalg.inv(
            np.insert(T_velo_cam, 3, values=[0, 0, 0, 1], axis=0)
        )

        # Store intrinsics and extrinsic inverse for 3-D back-projection.
        # T_ref0_velo: camera reference frame → LiDAR frame (4×4).
        self.K          = p_cam[:, :3]              # 3×3 camera intrinsics
        self.T_ref0_velo = np.linalg.inv(T_velo_ref0)  # cam ref → LiDAR

        return T_velo_cam, T_cam_velo

    # ── Main callback ─────────────────────────────────────────────────────────

    def callback(self, det_msg: DetectionArray, img_msg: Image, pcl_msg: PointCloud2):
        """
        Process one synchronised triplet: detections + image + LiDAR scan.

        Args:
            det_msg : DetectionArray from detector_node
            img_msg : Decoded source Image from detector_node
            pcl_msg : PointCloud2 from LiDAR driver
        """
        rospy.logdebug(
            "[fusion] Synchronised — image: %s  lidar: %s",
            img_msg.header.stamp, pcl_msg.header.stamp,
        )

        image    = imgmsg_to_cv2(img_msg)
        img_h, img_w = image.shape[:2]

        # ── Project LiDAR → camera frame (all points, fast numpy reader) ─────
        velo_uvz = self._project_pcl(pcl_msg, img_h, img_w)

        # ── Build detection numpy array ───────────────────────────────────────
        detections = det_msg.detections
        if not detections:
            return

        det_rows, det_names = [], []
        for d in detections:
            det_rows.append([d.xmin, d.ymin, d.xmax, d.ymax, d.confidence, 0])
            det_names.append(d.class_name)
        det_np = np.array(det_rows, dtype=np.float64)

        # ── LiDAR depth fusion for perception topics ──────────────────────────
        try:
            if velo_uvz is not None:
                det_with_depth = self._get_uvz_centers(image, velo_uvz, det_np, draw=False)
            else:
                raise ValueError("Empty LiDAR scan")
        except Exception as exc:
            logging.error("[fusion] LiDAR fusion failed — using depth=0: %s", exc)
            zeros = np.zeros((det_np.shape[0], 3), dtype=np.float64)
            det_with_depth = np.hstack((det_np, zeros))

        # ── Route to perception topics ────────────────────────────────────────
        self._route_and_publish(det_with_depth, det_names, img_w)

    # ── LiDAR fusion helper ───────────────────────────────────────────────────

    def _get_uvz_centers(
        self,
        image:     np.ndarray,
        velo_uvz:  tuple,
        bboxes:    np.ndarray,
        draw:      bool = True,
    ) -> np.ndarray:
        """
        Associate each bounding box with the nearest LiDAR point and append
        (u, v, z) depth coordinates.

        Args:
            image    : BGR image (modified in-place when draw=True).
            velo_uvz : (u, v, z) arrays of LiDAR points in camera space.
            bboxes   : (N, 6) array — [xmin, ymin, xmax, ymax, conf, cls].
            draw     : Overlay depth text on *image* when True.

        Returns:
            (N, 9) ndarray — original 6 columns plus (u, v, z).
        """
        from types import SimpleNamespace
        u, v, z = velo_uvz
        n = bboxes.shape[0]

        out = np.zeros((n, bboxes.shape[1] + 3), dtype=np.float64)
        out[:, :bboxes.shape[1]] = bboxes

        for i, bbox in enumerate(bboxes):
            obj_row_center = (bbox[1] + bbox[3]) / 2   # vertical midpoint
            obj_col_center = (bbox[0] + bbox[2]) / 2   # horizontal midpoint

            # Robust depth via inner-box histogram mode (same estimator as overlay).
            d_ns       = SimpleNamespace(xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3])
            z_robust   = self._estimate_depth(u, v, z, d_ns) or 0.0
            # Use box centre as the image-plane representative point so that
            # back-projection via _backproject_to_lidar aims at the object centre.
            out[i, -3:] = [obj_col_center, obj_row_center, z_robust]

            if draw:
                cx = int(round(obj_col_center))
                cy = int(round(obj_row_center))
                cv2.putText(
                    image,
                    f"{z_robust:.2f} m",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        return out

    # ── Perception message routing ────────────────────────────────────────────

    def _route_and_publish(self, det_with_depth: np.ndarray, class_names: list[str], img_w: int = 1920):
        """
        Group detections by class, build typed perception messages, and publish.

        det_with_depth columns: [xmin, ymin, xmax, ymax, conf, cls, u, v, z]

        Edge-clipped detections (box touching the left or right image boundary)
        are dropped here — their true object centre is off-screen so any
        position derived from them is unreliable.  Depth estimation on the
        overlay image is unaffected (that path is independent).
        """
        EDGE_MARGIN = 10  # px
        grouped: dict[str, list] = defaultdict(list)
        for row, name in zip(det_with_depth, class_names):
            key = get_topic_key(name)
            if key is None:
                continue
            xmin, xmax = row[0], row[2]
            if xmin <= EDGE_MARGIN or xmax >= img_w - EDGE_MARGIN:
                rospy.logdebug(
                    "[fusion] Dropping edge-clipped %s (xmin=%.0f xmax=%.0f)", name, xmin, xmax
                )
                continue
            grouped[key].append((row, name))

        if "objects" in grouped:
            self.pub_objects.publish(
                self._build_objects(grouped["objects"])
            )
        if "trafficsigns" in grouped:
            self.pub_trafficsigns.publish(
                self._build_trafficsigns(grouped["trafficsigns"])
            )
        if "trafficsignalhead" in grouped:
            self.pub_sighead.publish(
                self._build_trafficsignalhead(grouped["trafficsignalhead"])
            )
        if "lanemarkings" in grouped:
            self.pub_lane.publish(
                self._build_lanemarkings(grouped["lanemarkings"])
            )
        if "limitlines" in grouped:
            self.pub_limit.publish(
                self._build_limitlines(grouped["limitlines"])
            )
        if "road_state" in grouped:
            self.pub_road.publish(
                self._build_road_state(grouped["road_state"])
            )

    # ── Classifier cache callbacks ────────────────────────────────────────────

    def _cb_signs(self, msg):
        self._latest_signs = list(msg.signs)
        self._signs_stamp  = rospy.Time.now()

    def _cb_lights(self, msg):
        self._latest_lights = list(msg.lights)
        self._lights_stamp  = rospy.Time.now()

    def _cb_speeds(self, msg):
        self._latest_speeds = list(msg.speed_limits)
        self._speeds_stamp  = rospy.Time.now()

    def _cb_da(self, msg):
        self._da_mask = imgmsg_to_cv2(msg, encoding='mono8')

    def _cb_ll(self, msg):
        self._ll_mask = imgmsg_to_cv2(msg, encoding='mono8')

    def _match_classified(self, row, classified_list, max_dist: float = 80.0):
        """Return the classified item whose box centre is closest to row's box centre.

        Args:
            row:              Detection row [xmin, ymin, xmax, ymax, ...].
            classified_list:  List of ClassifiedSign or ClassifiedLight objects.
            max_dist:         Pixel radius beyond which no match is accepted.

        Returns:
            The best-matching item, or None if the list is empty or the closest
            item is farther than max_dist pixels.
        """
        det_cx = (row[0] + row[2]) / 2.0
        det_cy = (row[1] + row[3]) / 2.0
        best, best_dist = None, float("inf")
        for item in classified_list:
            cx = (item.xmin + item.xmax) / 2.0
            cy = (item.ymin + item.ymax) / 2.0
            dist = ((cx - det_cx) ** 2 + (cy - det_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = item
        return best if best_dist <= max_dist else None

    # ── Message builders ──────────────────────────────────────────────────────

    def _build_objects(self, items: list) -> Objects:
        """Build an Objects message from detections routed to 'objects'."""
        now           = datetime.now()
        hours_seconds = now.minute * 60   # matches old node behaviour

        objects = Objects()
        objects.Observation_Time_of_Hour = hours_seconds

        for idx, (row, name) in enumerate(items, start=1):
            xmin, ymin, xmax, ymax, conf = row[0], row[1], row[2], row[3], row[4]
            u_lidar, v_lidar, z_cam = float(row[-3]), float(row[-2]), float(row[-1])

            track = Objects_Track()
            track.ObjObjectIdA      = idx
            track.ObjObjectIdB      = idx
            track.ObjObjectIdC      = idx
            track.ObjObjectIdD      = idx
            track.ObjObjectType     = get_object_type(name)
            track.Confidence        = categorize_confidence(conf)
            track.Width             = xmax - xmin
            track.Height            = ymax - ymin
            track.Object_Source_Camera = 1

            # LongPos = forward distance (x in LiDAR frame, metres)
            # LatPos  = lateral offset  (y in LiDAR frame, metres)
            if z_cam > 0.0:
                long_pos, lat_pos = self._backproject_to_lidar(u_lidar, v_lidar, z_cam)
                track.LongPos = long_pos
                track.LatPos  = lat_pos
                track.Object_Source_Lidar = 1

            objects.objects_tracks.append(track)

        objects.Current_Object_Count = len(items)
        return objects

    def _build_trafficsigns(self, items: list) -> TrafficSigns:
        """Build a TrafficSigns message from sign detections.

        Matches each detection to the latest ClassifiedSign by bounding-box
        centre proximity, populates Sign_Type, Confidence, and LongPos/LatPos.
        """
        msg = TrafficSigns()
        for idx, (row, _) in enumerate(items, start=1):
            u_c, v_c, z_cam = float(row[-3]), float(row[-2]), float(row[-1])

            track = TrafficSign_Track()
            track.SignObjectID = idx

            classified = self._match_classified(row, self._latest_signs)
            if classified:
                track.Sign_Type  = _SIGN_TYPE_MAP.get(classified.sign_type, 0)
                track.Confidence = categorize_confidence(classified.type_confidence)

            if z_cam > 0.0:
                long_pos, lat_pos = self._backproject_to_lidar(u_c, v_c, z_cam)
                track.LongPos = long_pos
                track.LatPos  = lat_pos

            msg.trafficSign_track.append(track)

        msg.Current_Sign_Count       = len(items)
        msg.Observation_Time_of_Hour = datetime.now().minute * 60
        return msg

    def _build_trafficsignalhead(self, items: list) -> TrafficSignalsHead:
        """Build a TrafficSignalsHead message from traffic-light detections.

        Matches each detection to the latest ClassifiedLight by bounding-box
        centre proximity, sets the correct illumination field (e.g. IllumLtRedBall),
        and populates LongPos/LatPos from LiDAR back-projection.
        """
        msg = TrafficSignalsHead()
        for idx, (row, _) in enumerate(items, start=1):
            u_c, v_c, z_cam = float(row[-3]), float(row[-2]), float(row[-1])

            track = TrafficSignalsHead_Track()
            track.SignalObjectID    = idx
            track.Signal_Head_Type = 1   # 1 = standard traffic light

            classified = self._match_classified(row, self._latest_lights)
            if classified:
                track.Confidence = categorize_confidence(classified.state_confidence)
                illum_field = _LIGHT_ILLUM_MAP.get(classified.state)
                if illum_field:
                    setattr(track, illum_field, 1)

            if z_cam > 0.0:
                long_pos, lat_pos = self._backproject_to_lidar(u_c, v_c, z_cam)
                track.LongPos = long_pos
                track.LatPos  = lat_pos

            msg.trafficSignalsHead_track.append(track)

        msg.Current_Signal_Head_Count = len(items)
        msg.Observation_Time_of_Hour  = datetime.now().minute * 60
        return msg

    def _build_lanemarkings(self, items: list) -> LaneMarkings:
        """Build a LaneMarkings message."""
        msg = LaneMarkings()
        for _ in items:
            msg.laneMarking_track.append(LaneMarking_Track())
        return msg

    def _build_limitlines(self, items: list) -> LimitLines:
        """Build a LimitLines message."""
        msg = LimitLines()
        for _ in items:
            msg.limitline_track.append(LimitLine_Track())
        return msg

    def _build_road_state(self, items: list) -> road_state:
        """Build a road_state message."""
        return road_state()


if __name__ == "__main__":
    rospy.init_node("perception_fusion", anonymous=True)
    node = PerceptionFusionNode()
    rospy.spin()
