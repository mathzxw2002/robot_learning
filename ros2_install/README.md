1, Install and manage ubuntu on WSL2

wsl --list --online


列出当前安装的 WSL 分发版
wsl --list --verbose
# 输出示例:
# NAME            STATE           VERSION
# Ubuntu-20.04    Running         2
# Ubuntu-22.04    Stopped         2

停止目标分发版
在注销前，需要先停止运行中的分发版：
wsl --terminate <分发版名称>
# 例如: wsl --terminate Ubuntu-20.04

注销分发版
使用以下命令彻底删除分发版：

wsl --unregister <分发版名称>
# 例如: wsl --unregister Ubuntu-20.04

wsl --install Ubuntu


0, prepare a clean ubuntu

wsl 

sudo vim /etc/apt/sources.list.d/ubuntu.sources

URIs: http://mirrors.aliyun.com/ubuntu/

sudo apt update


ros2 install
(https://www.youtube.com/watch?v=HJAE5Pk8Nyw&ab_channel=KevinWood%7CRobotics%26AI)

1, install ubuntu on wsl2

https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/

2, install ros2

https://docs.ros.org/en/rolling/
https://docs.ros.org/en/rolling/Installation/Ubuntu-Install-Debs.html

3, source /opt/ros/rolling/setup.bash

4, https://docs.ros.org/en/rolling/Tutorials.html



5, install gazebo from source

https://gazebosim.org/docs/latest/install_ubuntu_src/


6, install nav2 from source

sudo rosdep init

rosdep update

pip3 install catkin_pkg empy lark-parser

pip3 install numpy


    
Once your ROS2 environment is setup, clone the repo and build the workspace:

source /opt/ros/rolling/setup.bash

mkdir -p ~/nav2_ws/src && cd ~/nav2_ws

git clone https://github.com/ros-navigation/navigation2.git --branch main ./src/navigation2

git clone https://github.com/ros-navigation/nav2_minimal_turtlebot_simulation.git --branch main ./src/nav2_minimal_turtlebot_simulation

rosdep install -r -y --from-paths ./src --ignore-src 

colcon build --symlink-install

You can then source ~/nav2_ws/install/setup.bash to get ready for demonstrations! It is safe to ignore the rosdep error of from the missing slam_toolbox key.

source ~/nav2_ws/install/setup.bash

sudo apt install ros-rolling-turtlebot3*

export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/rolling/share/turtlebot3_gazebo/models

ros2 launch nav2_bringup tb3_simulation_launch.py

5, slam_toolbox

mkdir -p ~/slam_toolbox_ws/src

cd ~/slam_toolbox_ws

git clone https://github.com/SteveMacenski/slam_toolbox







