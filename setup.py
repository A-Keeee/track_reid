from setuptools import setup

package_name = 'track_torch_ros2'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='ROS2 interface for track_torch pose publishing',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_publisher = track_torch_ros2.ros2_pose_publisher:main',
            'simple_pose_publisher = track_torch_ros2.simple_pose_publisher:main',
            'pose_subscriber = track_torch_ros2.pose_subscriber:main',
            'vision_control_subscriber = track_torch_ros2.vision_control_subscriber:main',
        ],
    },
)
