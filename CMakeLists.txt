# Mínimo requerido para un paquete de Python con scripts

cmake_minimum_required(VERSION 3.0.2)
project(acoustic_postprocessing)

# Buscar los paquetes que necesita este paquete
find_package(catkin REQUIRED COMPONENTS 
  rospy 
  std_msgs 
  ros_numpy
  # Añade aquí otros paquetes ROS que uses si son necesarios en C++
)

# Configurar el paquete para Python
catkin_python_setup()

# Instalar Scripts (Ejecutables)
# Esto le dice a Catkin que haga el script ejecutable y lo mueva a 'devel/lib/acoustic_postprocessing/'
catkin_python_setup()

# Instalar archivos de cabecera si los hubiera (no aplica aquí)
# install(DIRECTORY launch/
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
