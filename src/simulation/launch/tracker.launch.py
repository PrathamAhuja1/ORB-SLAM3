# tracker.launch.py
import os

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # If you want to use simulation time (Gazebo clock), uncomment:
    # use_sim_time = True
    return LaunchDescription([
        Node(
            package='simulation',
            executable='tracker_node',   # must match your setup.py entry_point
            name='orb_tracker',
            output='screen',
            # parameters=[{'use_sim_time': use_sim_time}],
            # remappings=[('image', '/stereo_left/image')],  # if you need to remap
        ),
    ])

