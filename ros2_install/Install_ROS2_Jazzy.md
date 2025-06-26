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
sudo apt install ros-jazzy-desktop

# Setup environment
source /opt/ros/jazzy/setup.bash

# Install turtlesim for demo
sudo apt update
sudo apt install ros-jazzy-turtlesim
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


# 4, install nav2 

```
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup

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

```
sudo apt install ros-$ROS_DISTRO-slam-toolbox

```


```
mkdir -p ~/slam_toolbox_ws/src
cd ~/slam_toolbox_ws/src
git clone https://github.com/SteveMacenski/slam_toolbox

colcon build --packages-select slam_toolbox --cmake-args -DCMAKE_BUILD_TYPE=Release

source ~/slam_toolbox_ws/install/setup.bash

```


# 6, bttree

```
sudo apt update
sudo apt install -y cmake g++ git libxml2-dev


mkdir -p ~/bt_ws/src
cd ~/bt_ws/src
git clone https://github.com/BehaviorTree/BehaviorTree.CPP.git

cd ~/bt_ws/src/BehaviorTree.CPP
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
```

# 7, robot-localization

```
sudo apt-get install ros-jazzy-robot-localization

```



