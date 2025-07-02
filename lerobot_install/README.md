
https://huggingface.co/docs/lerobot/lekiwi



run powershell as admin

usbipd wsl list



 winget install --interactive --exact dorssel.usbipd-win

 winget install dorssel.usbipd-win 

 # 检查版本（需 >= v0.5.0） 
  usbipd version  

  usbipd --list 

  usbipd bind --busid 2-2  

  # 语法：usbipd attach --busid <BUSID> --wsl-distribution <发行版名称>  


  usbipd attach --busid 2-2 --wsl Ubuntu-24.04 

  
