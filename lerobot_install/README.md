

# 在WSL2环境上，将USB连接到PC上

以管理员权限运行Windows PowerShell

```
# 安装 usbipd 
winget install --interactive --exact dorssel.usbipd-win
winget install dorssel.usbipd-win

# 检查版本（需 >= v0.5.0）
usbipd --version  

```


```
usbipd list
usbipd bind --busid 2-2
# 语法：usbipd attach --busid <BUSID> --wsl-distribution <发行版名称>  
usbipd attach --busid 2-2 --wsl ROS2Jazzy

python -m lerobot.find_port


```

https://huggingface.co/docs/lerobot/lekiwi


# Install LeRobot locally

Configure motors

https://gitee.com/ftservo/fddebug




# Install LeRobot on Pi 

```
ssh 192.168.0.232
```
![screenshot-20250709-113309](https://github.com/user-attachments/assets/37d84162-9c6a-4239-bf1e-d1b1ce5e49f7)

```
git clone https://github.com/huggingface/lerobot.git
cd lerobot


pip install -e .

pip install -e ".[aloha]" # or "[pusht]" for example

pip install -e ".[feetech]" # or "[dynamixel]" for example


```


# Calibration
```
Calibrate follower arm (on mobile base)

python -m lerobot.calibrate --robot.type=lekiwi --robot.id=my_awesome_kiwi # <- Give the robot a unique name


Calibrate leader arm

python -m lerobot.calibrate --teleop.type=so100_leader --teleop.port=/dev/tty --teleop.id=my_awesome_leader_arm # <- Give the robot a unique name



Teleoperate LeKiwi

To teleoperate, SSH into your Raspberry Pi, and run conda activate lerobot and this command:

Copied
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi
Then on your laptop, also run conda activate lerobot and run the API example, make sure you set the correct remote_ip and port in examples/lekiwi/teleoperate.py.

Copied
python examples/lekiwi/teleoperate.py


