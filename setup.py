## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Genera los metadatos necesarios para que Catkin sepa que es un paquete Python
d = generate_distutils_setup(
    # La lista de carpetas que contienen módulos/librerías Python.
    # Si tus scripts principales están en 'scripts/', puedes omitir 'packages'
    # packages=['pointcloud_lib'], 
    # package_dir={'': 'src'}
)

setup(**d)