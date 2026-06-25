# Mínimo requerido para un paquete de Python con scripts

cmake_minimum_required(VERSION 3.0.2)
project(acoustic_postprocessing)

# Buscar los paquetes que necesita este paquete. Debe coincidir con los
# <depend>/<buildtool_depend> de package.xml.
find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  ros_numpy
)

# Exporta este paquete a otros (no expone librerías C++; solo metadatos catkin).
catkin_package()

# Configura el paquete para Python (lee setup.py / genera metadatos catkin).
catkin_python_setup()

# Instala los scripts como ejecutables en
# 'lib/acoustic_postprocessing/' para que rosrun/roslaunch los encuentren
# también tras 'catkin_make install'.
catkin_install_python(PROGRAMS
  scripts/multibeam_slam.py
  scripts/make_presentation_media.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
