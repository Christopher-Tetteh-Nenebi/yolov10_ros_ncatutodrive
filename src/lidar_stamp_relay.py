#!/usr/bin/env python3
"""
LiDAR Timestamp Relay
=====================
Republishes a PointCloud2 topic with the header stamp shifted by a fixed
offset (default −4.672 s) so that the corrected topic aligns with the
camera stream for ApproximateTimeSynchronizer.

Usage
-----
  rosrun yolov10_ros lidar_stamp_relay.py

ROS Parameters
--------------
  ~input_topic   (str)   Source PointCloud2 topic.
                         Default: /synced/lidar/points
  ~output_topic  (str)   Republished topic with corrected stamps.
                         Default: /lidar/points_corrected
  ~offset_sec    (float) Seconds to ADD to the incoming stamp.
                         Use a negative value to shift backwards.
                         Default: -4.672
"""

import rospy
from sensor_msgs.msg import PointCloud2


def main():
    rospy.init_node("lidar_stamp_relay", anonymous=False)

    input_topic  = rospy.get_param("~input_topic",  "/synced/lidar/points")
    output_topic = rospy.get_param("~output_topic", "/lidar/points_corrected")
    offset_sec   = rospy.get_param("~offset_sec",   -4.672)

    offset = rospy.Duration.from_sec(offset_sec)
    pub    = rospy.Publisher(output_topic, PointCloud2, queue_size=1)

    def callback(msg: PointCloud2):
        msg.header.stamp += offset
        pub.publish(msg)

    rospy.Subscriber(input_topic, PointCloud2, callback, queue_size=1)
    rospy.loginfo(
        "[lidar_relay] %s → %s  (offset %.3f s)",
        input_topic, output_topic, offset_sec,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
