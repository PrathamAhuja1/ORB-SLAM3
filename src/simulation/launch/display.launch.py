import launch
import yaml
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import launch_ros.descriptions
import os
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package='simulation').find('simulation')
    default_model_path = os.path.join(pkg_share, 'src/description/orca4_description.urdf')
    default_rviz_config_path = os.path.join(pkg_share, 'rviz/urdf_config.rviz')
    world_path = os.path.join(pkg_share, "worlds/underwater_world_ign.sdf")
    gz_models_path = os.path.join(pkg_share, "models")
    bridge_config_path = os.path.join(pkg_share, 'config', 'ign_bridge.yaml')


    gz_verbosity = LaunchConfiguration("gz_verbosity")
    run_headless = LaunchConfiguration("run_headless")
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': launch_ros.descriptions.ParameterValue( launch.substitutions.Command(['xacro ', LaunchConfiguration('model')]) , value_type=str) } , {'use_sim_time': use_sim_time}]
    )
    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'robot_description': launch_ros.descriptions.ParameterValue( launch.substitutions.Command(['xacro ', default_model_path]) , value_type=str)}],
    )
    joint_state_publisher_gui_node = launch_ros.actions.Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=launch.conditions.IfCondition(LaunchConfiguration('gui'))
    )
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )

    gz_env = {'GZ_SIM_SYSTEM_PLUGIN_PATH':
           ':'.join([os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', default=''),
                     os.environ.get('LD_LIBRARY_PATH', default='')]),
           'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':  
                      ':'.join([os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', default=''),
                                os.environ.get('LD_LIBRARY_PATH', default='')])}
    
    ign_gazebo = [
        launch.actions.ExecuteProcess(
            condition=launch.conditions.IfCondition(run_headless),
            cmd=['ruby', launch.substitutions.FindExecutable(name="ign"), 'gazebo',  '-r', '-v', gz_verbosity, '-s', '--headless-rendering', world_path],
            output='screen',
            additional_env=gz_env,
            shell=False,
        ),
        launch.actions.ExecuteProcess(
            condition=launch.conditions.UnlessCondition(run_headless),
            cmd=['ruby', launch.substitutions.FindExecutable(name="ign"), 'gazebo',  '-r', '-v', gz_verbosity, world_path],
            output='screen',
            additional_env=gz_env,
            shell=False,
        )

    ]

    gz_sim = [

        launch.actions.ExecuteProcess(
            condition=launch.conditions.IfCondition(run_headless),
            cmd=[launch.substitutions.FindExecutable(name="gz"), 'sim',  '-r', '-v', gz_verbosity, '-s', '--headless-rendering', world_path],
            output='screen',
            additional_env=gz_env, 
            shell=False,
        ),
        launch.actions.ExecuteProcess(
            condition=launch.conditions.UnlessCondition(run_headless),
            cmd=[launch.substitutions.FindExecutable(name="gz"), 'sim',  '-r', '-v', gz_verbosity, world_path],
            output='screen',
            additional_env=gz_env, 
            shell=False,
        )
    ]

    spawn_entity = launch_ros.actions.Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name",
            "orca4_ign",
            "-topic",
            "robot_description",
            "-z",
            "2.5",
            "-x",
            "-4.0",
            "--ros-args",
            "--log-level",
            log_level,
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    bridge = launch_ros.actions.Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
                '--ros-args',
                '-p', f'config_file:={bridge_config_path}'
        ],
        output="screen",
    )

    bridge_old = launch_ros.actions.Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
            "/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU",
            "/stereo_left@sensor_msgs/msg/Image@gz.msgs.Image",
            "/stereo_right@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ],
        output="screen",
    )

    camera_bridge_image = launch_ros.actions.Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='bridge_gz_ros_camera_image',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['/stereo_left/image'],
    )

    camera_bridge_depth = launch_ros.actions.Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='bridge_gz_ros_camera_depth',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['/stereo_left/depth_image'],
    )

    return launch.LaunchDescription([   
        launch.actions.DeclareLaunchArgument(name='gui', default_value='True',
                                            description='Flag to enable joint_state_publisher_gui'),
        launch.actions.DeclareLaunchArgument(name='model', default_value=default_model_path,
                                            description='Absolute path to robot urdf file'),
        launch.actions.DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path,
                                            description='Absolute path to rviz config file'),
        launch.actions.DeclareLaunchArgument(
                name="use_sim_time",
                default_value="True",
                description="Flag to enable use_sim_time",
            ),
        launch.actions.DeclareLaunchArgument(
                name="log_level",
                default_value="warn",
                description="The level of logging that is applied to all ROS 2 nodes launched by this script.",
            ),
        launch.actions.DeclareLaunchArgument(
                name="run_headless",
                default_value="False",
                description="Start GZ in hedless mode and don't start RViz (overrides use_rviz)",
            ),
        launch.actions.DeclareLaunchArgument(
                "gz_verbosity",
                default_value="2",
                description="Verbosity level for gz sim (0~4).",
            ),
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz_node,
        bridge,
        # bridge_old,
        # camera_bridge_image,
        # camera_bridge_depth,
        # launch.actions.ExecuteProcess(cmd=['gz', 'sim', '-g'], output='screen'),
        spawn_entity,
    ] + ign_gazebo )