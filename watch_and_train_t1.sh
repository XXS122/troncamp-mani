#!/usr/bin/env bash
# watch_and_train_t1.sh [gpu_id] [batch_size] —— 监听 T1 采集进度,采满自动转换 + 训练。
#
#   每 60s 数一次 data/adjust_bottle/adjust_bottle_200ep/data/ 下的 episode*.hdf5;
#   到 200 个后:
#     1) process_data.sh 转 ACT 16-D 训练格式(纯 CPU,几分钟)
#     2) 直接调 imitate_episodes.py 开训(超参对齐 turnkey train.sh,仅 batch_size 可调,
#        ckpt 输出目录也与 turnkey 一致 → act_ckpt/act-adjust_bottle/adjust_bottle_200ep-200/,
#        后续 eval_local.py / submit.py 的路径不变)
#
# 用法:
#   nohup bash watch_and_train_t1.sh 0 4 > logs/watch_train_t1.log 2>&1 &
#                                    ^gpu ^batch_size(本机内存紧,默认 4)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

GPU="${1:-0}"
BATCH="${2:-4}"

TASK=adjust_bottle
CFG=adjust_bottle_200ep
NUM=200
DATA_DIR="external/robotwin_local/data/${TASK}/${CFG}/data"
ACT_DIR="external/robotwin_local/policy/ACT"
CKPT_DIR="${ACT_DIR}/act_ckpt/act-${TASK}/${CFG}-${NUM}"

echo "[watch] 开始监听 ${DATA_DIR},目标 ${NUM} 集(每 60s 查一次)"
while true; do
  n=$(ls "${DATA_DIR}"/episode*.hdf5 2>/dev/null | wc -l)
  echo "[watch] $(date '+%F %T') 当前 ${n}/${NUM}"
  [ "$n" -ge "$NUM" ] && break
  sleep 60
done
echo "[watch] 采集已达 ${NUM} 集,进入转换 + 训练"

# ---- 安全门:8G 显存放不下「采集 + 训练」同跑,必须等采集进程退出 ----
# AUTO_STOP_COLLECT=1 时自动停掉采集(collect_all.sh 重跑可断点续采,零损失);
# 默认只等待,由你手动决定何时停。
if pgrep -f collect_tron2_data.py > /dev/null; then
  if [ "${AUTO_STOP_COLLECT:-0}" = "1" ]; then
    echo "[gate] AUTO_STOP_COLLECT=1:停止采集进程(seed.txt 已落盘,重跑 collect_all.sh 即续采)"
    pkill -f collect_tron2_data.py || true
    pkill -f "bash collect_all.sh" || true
    sleep 10
  else
    echo "[gate] 检测到采集进程仍在跑。8G 显存无法同时训练,等待其退出…"
    echo "[gate] 想现在就训:手动执行 pkill -f collect_tron2_data.py(断点无损,训完重跑 collect_all.sh)"
    while pgrep -f collect_tron2_data.py > /dev/null; do sleep 60; done
    echo "[gate] 采集进程已退出,继续。"
  fi
fi

# ---- 1) 转 ACT 训练格式(已转过则跳过:processed_data 齐 200 个 episode_*.hdf5 即认为完成) ----
PROCESSED="${ACT_DIR}/processed_data/sim-${TASK}/${CFG}-${NUM}"
np=$(ls "${PROCESSED}"/episode_*.hdf5 2>/dev/null | wc -l)
if [ "$np" -ge "$NUM" ]; then
  echo "[process] 已存在 ${np} 个转换后 episode,跳过转换"
else
  ( cd "${ACT_DIR}" && bash process_data.sh "${TASK}" "${CFG}" "${NUM}" )
fi

# ---- 2) 训练(超参与 turnkey train.sh 一致,仅 batch_size=${BATCH};单卡) ----
echo "[train] 开始训练:batch_size=${BATCH},GPU ${GPU},ckpt → ${CKPT_DIR}"
cd "${ACT_DIR}"
export CUDA_VISIBLE_DEVICES=${GPU}
python3 imitate_episodes.py \
    --task_name sim-${TASK}-${CFG}-${NUM} \
    --ckpt_dir ./act_ckpt/act-${TASK}/${CFG}-${NUM} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size ${BATCH} \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 2000 \
    --state_dim 16 \
    --seed 0

echo "[train] 完成。下一步本地自评:"
echo "  python starter/eval_local.py --track T1 --ckpt-dir ${CKPT_DIR}"
