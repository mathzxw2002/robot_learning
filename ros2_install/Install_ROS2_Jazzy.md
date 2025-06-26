# 1, Install and manage ubuntu on WSL2

run wsl --list --online to see an output with all available distros and versions:
```
wsl --list --online
```

use wsl --install xx to install WSL system from the command line
```
wsl --install Ubuntu-24.04 --name ROS2Jazzy 
```
It is recommended to reboot your machine after this initial installation to complete the setup.

revise source list

```
sudo chmod 777 /etc/apt/sources.list.d/ubuntu.sources

sudo vim /etc/apt/sources.list.d/ubuntu.sources
```

replace origin URIs with URIs: http://mirrors.aliyun.com/ubuntu/

```
sudo apt update
```

# 2, ros2 install

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
sudo apt install ros-jazzy-desktop

# Setup environment
source /opt/ros/jazzy/setup.bash

```


# 3, install nav2 

```
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup

```


# 4, slam_toolbox

```
sudo apt install ros-jazzy-slam-toolbox

```

# 5, robot-localization

```
sudo apt-get install ros-jazzy-robot-localization

```

# 6, nav2 example 
https://docs.nav2.org/getting_started/index.html


```
sudo apt install ros-jazzy-turtlebot3*
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/jazzy/share/turtlebot3_gazebo/models
# Disable GPU hardware acceleration
export LIBGL_ALWAYS_SOFTWARE=1

ros2 launch nav2_bringup tb3_simulation_launch.py
```

