# install Ubuntu24.04
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04

RUN sed -i.bak 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    apt update && \
    apt upgrade -y && \
    apt install git -y && \
    apt install pip -y  && \
    apt install wget && \
    wget https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz && \
    mkdir -p ~/.mujoco && tar -zxf mujoco-2.3.7-linux-x86_64.tar.gz -C ~/.mujoco/ && \
    export MUJOCO_PATH="$HOME/.mujoco/mujoco-2.3.7" && export MUJOCO_PLUGIN_PATH="$MUJOCO_PATH/plugin" && export LD_LIBRARY_PATH="$MUJOCO_PATH/lib:$MUJOCO_PLUGIN_PATH:$LD_LIBRARY_PATH" && \
    pip install --no-cache-dir --force-reinstall mujoco && \
    git clone https://github.com/huggingface/lerobot.git && cd lerobot && \
    apt install -y ffmpeg && \
    apt install -y libx11-dev libxrandr-dev libxi-dev libxinerama-dev libxcursor-dev libgl1-mesa-dev && \
    pip install mujoco --break-system-packages && \
    pip install -e ".[aloha, pusht, lekiwi, feetech]" --break-system-packages   
