#!/bin/bash

# 1. 设置错误日志输出（如果执行失败，你可以去 /home/ubuntu/train_debug.log 查看原因）
exec > /home/ubuntu/train_debug.log 2>&1
set -x  # 开启调试模式，记录每一步执行过程

# 2. 初始化 Conda 环境变量（这是非交互式脚本能用 conda activate 的关键）
# 注意：请确保 /home/ubuntu/laiwenqi/anaconda3 是你的安装路径
CONDA_PATH="/home/ubuntu/laiwenqi/anaconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "Error: Conda profile.d script not found at $CONDA_PATH"
    exit 1
fi

# 3. 激活虚拟环境
conda activate IQ

# 4. 切换到项目目录，如果失败则退出，防止在错误目录下执行
cd /home/ubuntu/laiwenqi/projects/IQ-learning-constraint/iq_learn || { echo "Directory not found"; exit 1; }

# 5. 执行训练
# 这里直接用 python，因为上面已经 source 并 activate 了，系统会自动寻找 IQ 环境的 python
python train_iq.py \
    env=hopper \
    agent=sac \
    expert.demos=1 \
    expert.offline_demos=1 \
    offline=True \
    method.loss=dice \
    seed=0 \
    method.normal_r=true \
    method.constrain=true \
    method.div=kl

echo "Training task finished at $(date)"