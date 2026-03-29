#!/usr/bin/env python3
"""
YOLOv10 Detector Node
=====================
Thin inference node: subscribes to a camera topic, runs YOLOv10, and
publishes bounding-box detections plus the decoded source frame.

This node intentionally does *not* perform LiDAR fusion or downstream
perception-message routing — those responsibilities live in
``perception_fusion_node.py``, which subscribes to this node's output
topics and synchronises them with the LiDAR stream independently.

Subscriptions
-------------
  ~input_image_topic  (sensor_msgs/Image or sensor_msgs/CompressedImage)
      Raw camera frames.  Topic type is auto-detected at startup.

Publications
------------
  ~output_topic           (yolov10_ros/DetectionArray)
      Per-frame bounding boxes with class name and confidence.
      Default: /yolov10/detections

  ~output_image_topic     (sensor_msgs/Image)
      Decoded source frame re-published so downstream nodes can crop ROIs
      without their own camera subscriber.
      Default: /yolov10/image_raw

  ~annotated_image_topic  (sensor_msgs/Image)
      Debug view with boxes drawn.  Only published when
      ``publish_annotated`` is true.
      Default: /yolov10/annotated

ROS Parameters
--------------
  ~weights               (str)   Path to .pt weights file.
  ~confidence_threshold  (float) Minimum confidence to keep a detection.
                                 Default: 0.5
  ~inference_size        (int)   Model input resolution (square pixels).
                                 Default: 640
  ~device                (str)   CUDA device ("0", "1", …) or "cpu".
                                 Default: "0"
  ~half                  (bool)  FP16 inference (GPU only).
                                 Default: false
  ~input_image_topic     (str)   Camera topic to subscribe to.
  ~view_image            (bool)  Show OpenCV window.
                                 Default: false
  ~publish_annotated     (bool)  Publish annotated image topic.
                                 Default: false
"""

import rospy
import cv2
import torch
import numpy as np
from rostopic import get_topic_type

from sensor_msgs.msg import Image, CompressedImage
from yolov10_ros.image_utils import imgmsg_to_cv2, cv2_to_imgmsg
from yolov10_ros.visualization import draw_detections


class YOLOv10Detector:
    """
    Thin ROS wrapper around a YOLOv10 object detector.

    Decodes incoming frames, runs inference, and publishes a DetectionArray
    plus the source image so downstream nodes can work without their own
    camera subscriber.
    """

    def __init__(self):
        # ── Parameters ────────────────────────────────────────────────────────
        self.conf_thres      = rospy.get_param("~confidence_threshold", 0.5)
        self.device          = str(rospy.get_param("~device", "0"))
        self.inference_size  = rospy.get_param("~inference_size", 640)
        self.view_image      = rospy.get_param("~view_image", False)
        self.publish_annotated = rospy.get_param("~publish_annotated", False)
        self.half            = rospy.get_param("~half", False)
        weights              = rospy.get_param("~weights", "")

        # ── Model loading ─────────────────────────────────────────────────────
        # Try the pinned YOLOv10 class (ultralytics ≥ 8.1) and fall back to
        # the generic YOLO wrapper for older installs.  Both expose .predict().
        try:
            from ultralytics import YOLOv10
            self.model = YOLOv10(weights)
            rospy.loginfo("[detector] Loaded model with ultralytics.YOLOv10")
        except ImportError:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            rospy.loginfo("[detector] Loaded model with ultralytics.YOLO (fallback)")

        rospy.loginfo("[detector] Model classes: %s", self.model.names)

        # ── Subscriber ────────────────────────────────────────────────────────
        input_image_topic = rospy.get_param("~input_image_topic", "/camera/color/image_raw")
        input_image_type, input_image_topic, _ = get_topic_type(
            input_image_topic, blocking=True
        )
        self.compressed_input = (input_image_type == "sensor_msgs/CompressedImage")
        msg_type = CompressedImage if self.compressed_input else Image

        # queue_size=1 + large buff_size: always process the latest frame;
        # drop stale ones when inference is slower than camera rate.
        self.image_sub = rospy.Subscriber(
            input_image_topic, msg_type, self.callback,
            queue_size=1, buff_size=2 ** 24,
        )
        rospy.loginfo("[detector] Subscribed to %s [%s]", input_image_topic, input_image_type)

        # ── Publishers ────────────────────────────────────────────────────────
        # Lazy-import generated message types so the node starts before catkin
        # has compiled them in a fresh devel build.
        from yolov10_ros.msg import Detection, DetectionArray
        self.Detection      = Detection
        self.DetectionArray = DetectionArray

        output_topic = rospy.get_param("~output_topic", "/yolov10/detections")
        self.det_pub = rospy.Publisher(output_topic, DetectionArray, queue_size=10)

        output_image_topic = rospy.get_param("~output_image_topic", "/yolov10/image_raw")
        self.image_pub = rospy.Publisher(output_image_topic, Image, queue_size=1)

        if self.publish_annotated:
            annotated_topic = rospy.get_param("~annotated_image_topic", "/yolov10/annotated")
            self.annotated_pub = rospy.Publisher(annotated_topic, Image, queue_size=1)

    # ── Inference callback ────────────────────────────────────────────────────

    @torch.no_grad()
    def callback(self, msg):
        """
        Process one camera frame: decode → infer → publish.

        Args:
            msg: sensor_msgs/Image or sensor_msgs/CompressedImage.
        """
        # Decode to BGR numpy array (imgmsg_to_cv2 handles both types).
        im0 = imgmsg_to_cv2(msg)

        # Inference — ultralytics handles resize, letterbox, normalise, and
        # de-letterbox internally.  Returned boxes are in original pixel coords.
        results = self.model.predict(
            source=im0,
            conf=self.conf_thres,
            imgsz=self.inference_size,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        # ── Build DetectionArray ──────────────────────────────────────────────
        det_array = self.DetectionArray()
        det_array.header = msg.header   # preserve original timestamp for ApproxTimeSynch

        boxes = results[0].boxes       # Boxes object (None if no detections)
        names = results[0].names       # {class_idx: class_name}

        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                xyxy    = boxes.xyxy[i].cpu().numpy()
                conf    = float(boxes.conf[i].cpu())
                cls_idx = int(boxes.cls[i].cpu())

                det            = self.Detection()
                det.class_name = names[cls_idx]
                det.confidence = conf
                det.xmin       = int(xyxy[0])
                det.ymin       = int(xyxy[1])
                det.xmax       = int(xyxy[2])
                det.ymax       = int(xyxy[3])
                det_array.detections.append(det)

        # ── Publish ───────────────────────────────────────────────────────────
        self.det_pub.publish(det_array)

        # Republish the decoded source frame so the fusion node (and any
        # classifier or OCR node) can crop ROIs without their own camera sub.
        img_msg        = cv2_to_imgmsg(im0)
        img_msg.header = msg.header    # preserve timestamp for synchronisation
        self.image_pub.publish(img_msg)

        # Optional annotated image for rviz / rqt_image_view.
        if self.publish_annotated or self.view_image:
            annotated = draw_detections(im0, det_array.detections)
            if self.publish_annotated:
                self.annotated_pub.publish(cv2_to_imgmsg(annotated))
            if self.view_image:
                cv2.imshow("YOLOv10 Detections", annotated)
                cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node("yolov10_detector", anonymous=True)
    detector = YOLOv10Detector()
    rospy.spin()
