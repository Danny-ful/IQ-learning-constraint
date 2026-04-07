cd /home/ubuntu/laiwenqi/projects/IQ-learning-constraint/iq_learn

conda activate IQ

python train_iq.py env=hopper agent=sac expert.demos=1 expert.offline_demos=1 offline=True method.loss=dice seed=0 method.normal_r=true method.constrain=true method.div=kl