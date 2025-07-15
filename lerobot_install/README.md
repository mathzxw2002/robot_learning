# Lerobot Github
https://github.com/huggingface/lerobot


https://huggingface.co/docs/lerobot/lekiwi


# Install LeRobot locally on PC

## 1, Install Ubuntu-24.04 on WSL2

(more details about WSL2 command. https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/)

```
wsl --install -d Ubuntu-24.04 --name Lerobot
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
usbipd attach --busid 2-2 --wsl ROS2Jazz

Lerobot

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

# Evaluate our policy on Lerobot

To evaluate policy, SSH into your Raspberry Pi, and run conda activate lerobot and this command:

```
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi
```

To evaluate your policy run the evaluate.py API example, make sure to change remote_ip, port, model..

```

python examples/lekiwi/evaluate.py

```



# Bugs
## 1 
python -m lerobot.calibrate --robot.type=lekiwi --robot.id=my_awesome_kiwi
![screenshot-20250709-115522](https://github.com/user-attachments/assets/3d2ad2f6-916a-4e67-bb63-889994c8c3a4)
vim /home/china/lerobot/src/lerobot/robots/lekiwi/config_lekiwi.py
![image](https://github.com/user-attachments/assets/cbc00e78-9db2-4e96-81ed-07c73ee6185a)


CUDA_VISIBLE_DEVICES=""


## 2 stats missing when using pretrained pi0 policy #694

```
Traceback (most recent call last):
  File "/scratch/zf540/pi0/aqua-vla/experiments/envs/simpler/test_ckpts_in_simpler.py", line 222, in <module>
    eval_simpler()
  File "/scratch/zf540/pi0/aqua-vla/.venv/lib/python3.11/site-packages/draccus/argparsing.py", line 225, in wrapper_inner
    response = fn(cfg, *args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/zf540/pi0/aqua-vla/experiments/envs/simpler/test_ckpts_in_simpler.py", line 174, in eval_simpler
    action = policy.select_action(observation)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/zf540/pi0/aqua-vla/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/zf540/pi0/lerobot/lerobot/common/policies/pi0/modeling_pi0.py", line 276, in select_action
    batch = self.normalize_inputs(batch)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/zf540/pi0/aqua-vla/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/zf540/pi0/aqua-vla/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/zf540/pi0/aqua-vla/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/zf540/pi0/lerobot/lerobot/common/policies/normalize.py", line 155, in forward
    assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: `mean` is infinity. You should either initialize with `stats` as an argument, or use a pretrained model.

```

<img width="1406" height="755" alt="image" src="https://github.com/user-attachments/assets/eac92bd7-08df-498b-a515-dbd2131c3415" />



