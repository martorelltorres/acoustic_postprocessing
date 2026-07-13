# Acoustic post-processing package (Python)

cmake_minimum_required(VERSION 3.0.2)
project(acoustic_postprocessing)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  ros_numpy
)

# Python package metadata (setup.py)
catkin_python_setup()

catkin_package()

# Install executable scripts into lib/${PROJECT_NAME}
catkin_install_python(PROGRAMS
  scripts/multibeam_processor.py
  scripts/multibeam_intensity.py
  scripts/sss2mosaic.py
  scripts/sss_waterfall.py
  scripts/mb_sss_mosaic_fusion.py
  scripts/sss_mb_fusion.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
