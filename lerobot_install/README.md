

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
usbipd attach --busid 2-2 --wsl Ubuntu-24.04 

```

https://huggingface.co/docs/lerobot/lekiwi
