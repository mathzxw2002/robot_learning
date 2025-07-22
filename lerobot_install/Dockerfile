# install Ubuntu22.04
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i.bak "s@http://.*archive.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    apt update && \
    apt upgrade -y && \
    apt install -y cmake git pip wget ffmpeg libx11-dev libxrandr-dev libxi-dev libxinerama-dev libxcursor-dev libgl1-mesa-dev && \
    wget https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz && \
    mkdir -p ~/.mujoco && tar -zxf mujoco-2.3.7-linux-x86_64.tar.gz -C ~/.mujoco/ && \
    export MUJOCO_PATH="$HOME/.mujoco/mujoco-2.3.7" && export MUJOCO_PLUGIN_PATH="$MUJOCO_PATH/plugin" && export LD_LIBRARY_PATH="$MUJOCO_PATH/lib:$MUJOCO_PLUGIN_PATH:$LD_LIBRARY_PATH" && \
    pip install --no-cache-dir --force-reinstall mujoco --break-system-packages && \
    pip install numpy==2.2.0 --break-system-packages && pip install fsspec==2025.3.0 --break-system-packages && \
    git clone https://github.com/huggingface/lerobot.git && cd lerobot && \
    pip install -e ".[aloha, pusht, lekiwi, feetech]" --break-system-packages   
