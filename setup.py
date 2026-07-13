## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Generates the metadata Catkin needs to treat this as a Python package
d = generate_distutils_setup(
    # Folders holding Python modules. Can be omitted when the entry points
    # live in 'scripts/'.
    # packages=['pointcloud_lib'], 
    # package_dir={'': 'src'}
)

setup(**d)