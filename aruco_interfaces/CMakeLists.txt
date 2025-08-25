cmake_minimum_required(VERSION 3.8)
project(aruco_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(SRV_FILES
  "srv/ArucoPoseEstimate.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${SRV_FILES}
  DEPENDENCIES sensor_msgs geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime sensor_msgs geometry_msgs std_msgs)
ament_package()