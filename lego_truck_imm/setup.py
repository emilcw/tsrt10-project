from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
import rospy

setup_args = generate_distutils_setup(
    scripts=['bin/imm'],
    packages=['lego_truck_imm'],
    package_dir={'': 'src'}
)
setup(**setup_args)
