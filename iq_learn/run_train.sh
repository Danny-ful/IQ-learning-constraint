#!/bin/bash

# --- 1. 解决挂载延迟 ---
# 强制等待 15 秒，确保 /home/ubuntu 下的硬盘已经挂载成功

# --- 2. 解决路径和权限问题 ---
# 确保即使是 root 身份，也能强制使用 ubuntu 用户的环境
export USER=ubuntu
export HOME=/home/ubuntu
cd /home/ubuntu/laiwenqi/projects/IQ-learning-constraint/iq_learn || exit 1

# --- 3. 彻底初始化 Conda ---
# 不要依赖系统的 PATH，直接手动指认 conda.sh
CONDA_PROFILE="/home/ubuntu/laiwenqi/anaconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate IQ
else
    # 如果路径不对，尝试另一个可能的路径
    source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
    conda activate IQ
fi

# MuJoCo 2.1 + mujoco_py：需能找到 libmujoco210.so（见 ~/.mujoco/mujoco210/bin）
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/home/ubuntu/.mujoco/mujoco210/bin"

# --- 4. 运行程序 ---
# 加上全路径，确保万无一失
python train_iq.py \
    env=hopper \
    agent=sac \
    expert.demos=10 \
    expert.offline_demos=10 \
    offline=True \
    method.loss=dice \
    seed=0 \
    method.normal_r=true \
    method.div=kl \
    env.learn_steps=1000000
