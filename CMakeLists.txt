cmake_minimum_required(VERSION 3.0.2)
project(yolov10_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  Detection.msg
  DetectionArray.msg
  ClassifiedLight.msg
  ClassifiedLightArray.msg
  ClassifiedSign.msg
  ClassifiedSignArray.msg
  SpeedLimit.msg
  SpeedLimitArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs cv_bridge image_transport message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/detector_node.py
  src/light_classifier_node.py
  src/sign_classifier_node.py
  src/speed_limit_ocr_node.py
  src/perception_viz_node.py
  src/lane_seg_node.py
  src/perception_fusion_node.py
  src/scoring_can.py
  src/lidar_stamp_relay.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
