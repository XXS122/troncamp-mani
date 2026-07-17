#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

DEBUG=False
save_ckpt=True

# env 可覆盖(不传保持 turnkey 默认): NUM_EPOCHS / BATCH_SIZE。
# 另有训练提速旋钮(见 imitate_episodes.py / utils.py):
#   ACT_AMP=1(默认) bf16 混合精度; ACT_VAL_EVERY=5(默认) 验证间隔;
#   ACT_EARLY_STOP_PATIENCE=0(默认关) 早停; ACT_WORKERS=2(默认) dataloader 进程数。
num_epochs=${NUM_EPOCHS:-6000}
batch_size=${BATCH_SIZE:-8}

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./act_ckpt/act-${task_name}/${task_config}-${expert_data_num} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size ${batch_size} \
    --dim_feedforward 3200 \
    --num_epochs ${num_epochs} \
    --lr 1e-5 \
    --save_freq 2000 \
    --state_dim 16 \
    --seed ${seed}
