

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


git clone https://github.com/huggingface/lerobot.git
cd lerobot


```
