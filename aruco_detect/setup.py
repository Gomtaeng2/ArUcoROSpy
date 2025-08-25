import os
from glob import glob
from setuptools import setup

package_name = 'aruco_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_data={
        'aruco_detect': ['marker_transforms.npz'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jure',
    maintainer_email='jure@todo.todo',
    description='The aruco_detect package for ROS2',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_node = aruco_detect.aruco_node:main',
            'aruco_service = aruco_detect.aruco_service:main',
            'aruco_calibrate = aruco_detect.aruco_calibrate:main',
            'aruco_call_service = aruco_detect.aruco_call_service:main',
        ],
    },
)
