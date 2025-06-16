# 1, Install and manage ubuntu on WSL2

(https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/)

run wsl --list --online to see an output with all available distros and versions:
```
wsl --list --online
```

use wsl -l -v to see all your currently installed distros and the version of WSL that they are using:
```
wsl --list --verbose
```

use wsl --terminate xx to stop a system, and wsl --unregister xx to delete a system
```
wsl --terminate xxx

wsl --unregister xxx
```

use wsl --install xx to install WSL system from the command line
```
wsl --install xxx
```
It is recommended to reboot your machine after this initial installation to complete the setup.

# 2, revise source list
sudo chmod 777 /etc/apt/sources.list.d/ubuntu.sources

sudo vim /etc/apt/sources.list.d/ubuntu.sources

URIs: http://mirrors.aliyun.com/ubuntu/

sudo apt update


# 3, ros2 install
(https://www.youtube.com/watch?v=HJAE5Pk8Nyw&ab_channel=KevinWood%7CRobotics%26AI)


https://docs.ros.org/en/rolling/
https://docs.ros.org/en/rolling/Installation/Ubuntu-Install-Debs.html

```
# Set locale
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale  # verify settings

# Enable required repositories
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" # If using Ubuntu derivates use $UBUNTU_CODENAME
sudo apt install /tmp/ros2-apt-source.deb

# Install development tools (optional)
sudo apt update && sudo apt install ros-dev-tools

# Install ROS 2
sudo apt update
sudo apt upgrade
sudo apt install ros-rolling-desktop

# Setup environment
source /opt/ros/rolling/setup.bash

# Install turtlesim for demo
sudo apt update
sudo apt install ros-rolling-turtlesim
ros2 run turtlesim turtlesim_node

# Open a new terminal and source ROS 2 again. Now you will run a new node to control the turtle in the first node:
ros2 run turtlesim turtle_teleop_key

```

Try some examples
```
In one terminal, source the setup file and then run a C++ talker:
source /opt/ros/rolling/setup.bash

In another terminal source the setup file and then run a Python listener:
ros2 run demo_nodes_cpp talker

source /opt/ros/rolling/setup.bash

```

4, https://docs.ros.org/en/rolling/Tutorials.html


# 6, install nav2 from source

```
sudo rosdep init
rosdep update

pip3 install catkin_pkg empy lark-parser
pip3 install numpy


    
# Once your ROS2 environment is setup, clone the repo and build the workspace:

source /opt/ros/rolling/setup.bash
mkdir -p ~/nav2_ws/src && cd ~/nav2_ws
git clone https://github.com/ros-navigation/navigation2.git --branch main ./src/navigation2
git clone https://github.com/ros-navigation/nav2_minimal_turtlebot_simulation.git --branch main ./src/nav2_minimal_turtlebot_simulation
rosdep install -r -y --from-paths ./src --ignore-src 
colcon build --symlink-install
# You can then source ~/nav2_ws/install/setup.bash to get ready for demonstrations! It is safe to ignore the rosdep error of from the missing slam_toolbox key.
source ~/nav2_ws/install/setup.bash
```

```
sudo apt install ros-rolling-turtlebot3*
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/rolling/share/turtlebot3_gazebo/models
ros2 launch nav2_bringup tb3_simulation_launch.py
```
禁用 GPU 硬件加速
export LIBGL_ALWAYS_SOFTWARE=1

# 5, slam_toolbox

install message_filter first, which requires version 7.1.1

```
mkdir -p ~/message_filters_ws/src
cd ~/message_filters_ws/src
git clone https://github.com/ros2/message_filters.git -b kilted
colcon build --packages-select message_filters --cmake-args -DCMAKE_BUILD_TYPE=Release --allow-overriding message_filters

sudo cp -rf ~/message_filters_ws/install/message_filters/include/* /opt/ros/rolling/include/
sudo cp -rf ~/message_filters_ws/install/message_filters/lib/*.so /opt/ros/rolling/lib/
sudo cp -rf ~/message_filters_ws/install/message_filters/share/message_filters/* /opt/ros/rolling/share/message_filters
```


```
mkdir -p ~/slam_toolbox_ws/src
cd ~/slam_toolbox_ws/src
git clone https://github.com/SteveMacenski/slam_toolbox

colcon build --packages-select slam_toolbox --cmake-args -DCMAKE_BUILD_TYPE=Release

source ~/slam_toolbox_ws/install/setup.bash

```





