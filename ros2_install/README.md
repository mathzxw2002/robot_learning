ros2 install
(https://www.youtube.com/watch?v=HJAE5Pk8Nyw&ab_channel=KevinWood%7CRobotics%26AI)

1, install ubuntu on wsl2

https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/

2, install ros2

https://docs.ros.org/en/rolling/
https://docs.ros.org/en/rolling/Installation/Ubuntu-Install-Debs.html

3, source /opt/ros/rolling/setup.bash

4, https://docs.ros.org/en/rolling/Tutorials.html

5, install nav2 from source

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


6,




