#!/bin/bash

cd /home/ubuntu/laiwenqi/projects/IQ-learning-constraint/iq_learn || exit

PYTHON_EXEC="/home/ubuntu/laiwenqi/anaconda3/bin/python"

$PYTHON_EXEC train_iq.py \
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