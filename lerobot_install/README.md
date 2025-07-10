# Lerobot Github
https://github.com/huggingface/lerobot


https://huggingface.co/docs/lerobot/lekiwi


# Install LeRobot locally on PC

## 1, Install Ubuntu-24.04 on WSL2

(more details about WSL2 command. https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/)

```
wsl --install -d Ubuntu-24.04 
```
It is recommended to reboot your machine after this initial installation to complete the setup.

Then, revise source list,

```
sudo chmod 777 /etc/apt/sources.list.d/ubuntu.sources
sudo vim /etc/apt/sources.list.d/ubuntu.sources

replace original source list with aliyun version, i.e. URIs: http://mirrors.aliyun.com/ubuntu/

sudo apt update
```

## 2, Install lerobot on WSL2 Ubuntu
```
git clone https://github.com/huggingface/lerobot.git
cd lerobot

pip install -e .

```

## 3, 在WSL2环境上，将USB连接到PC上

以管理员权限运行Windows PowerShell

```
# 安装 usbipd 
winget install --interactive --exact dorssel.usbipd-win
winget install dorssel.usbipd-win

# 检查版本（需 >= v0.5.0）
usbipd --version  
usbipd list
usbipd bind --busid 2-2
# 语法：usbipd attach --busid <BUSID> --wsl-distribution <发行版名称>  
usbipd attach --busid 2-2 --wsl ROS2Jazzy

```

在WSL2 Ubuntu系统下，
```
cd lerobot
python -m lerobot.find_port

sudo chmod 666 /dev/ttyACM0

```


## 5, 配置Lekiwi舵机参数
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

Calibrate follower arm (on mobile base)
```
python -m lerobot.calibrate --robot.type=lekiwi --robot.id=my_awesome_kiwi # <- Give the robot a unique name
```

Calibrate leader arm
```
python -m lerobot.calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_awesome_leader_arm # <- Give the robot a unique name
```


Teleoperate LeKiwi

To teleoperate, SSH into your Raspberry Pi, and run conda activate lerobot and this command:

```
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi
```
Then on your laptop, also run conda activate lerobot and run the API example, make sure you set the correct remote_ip and port in examples/lekiwi/teleoperate.py.

```
python examples/lekiwi/teleoperate.py
```

/home/china/lerobot/examples/lekiwi/teleoperate.py
![image](https://github.com/user-attachments/assets/c93cb1c6-aef4-477c-ab5a-0893141f4ae5)


export LIBGL_ALWAYS_SOFTWARE=1

# Bugs
python -m lerobot.calibrate --robot.type=lekiwi --robot.id=my_awesome_kiwi
![screenshot-20250709-115522](https://github.com/user-attachments/assets/3d2ad2f6-916a-4e67-bb63-889994c8c3a4)
vim /home/china/lerobot/src/lerobot/robots/lekiwi/config_lekiwi.py
![image](https://github.com/user-attachments/assets/cbc00e78-9db2-4e96-81ed-07c73ee6185a)


# 建议将上面这一行写入 ~/.bashrc。若没有写入，则每次下载时都需要先输入该命令
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=""



