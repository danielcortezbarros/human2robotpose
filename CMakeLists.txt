cmake_minimum_required(VERSION 3.0.2)
project(human2robotpose)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

# add_message_files(
#   FILES
#   file.msg
# )

# add_service_files(
#   FILES
#   file.srv
# )

# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

catkin_package(
  CATKIN_DEPENDS rospy std_msgs message_runtime
)

install(PROGRAMS
  src/human2robotpose_application.py
  src/human2robotpose_implementation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
