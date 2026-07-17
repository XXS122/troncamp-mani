#!/usr/bin/env bash
# collect_all.sh [gpu_id] —— 四任务顺序自采(T1 → T2 → T3 → T4),单卡串行。
#
#   每个任务都走 collect_data.sh(断点续采:中断后重跑本脚本,已采够的任务
#   会读 seed.txt 秒过,未采完的从断点继续,不会重采)。
#   日志落在 logs/collect/<task>_<config>.log,终端同时可见(tee)。
#
# 用法:
#   bash collect_all.sh        # 默认 GPU 0
#   bash collect_all.sh 1      # 指定 GPU
#
# 后台挂起跑(推荐,防 ssh 断连):
#   nohup bash collect_all.sh 0 > logs/collect/all.log 2>&1 &
set -u
cd "$(dirname "$0")"

# 输出经 tee 走管道,Python 会切成块缓冲导致日志/终端长时间无输出、看似卡死;
# 强制不缓冲,恢复逐行实时输出。
export PYTHONUNBUFFERED=1

GPU="${1:-0}"
LOG_DIR="logs/collect"
mkdir -p "$LOG_DIR"

# 任务清单:<task_name> <task_config>(episode 数写在各自 yml 里:200/300/400/500)
TASKS=(
  "adjust_bottle      adjust_bottle_200ep"
  "grab_roller        grab_roller_300ep"
  "stack_bowls_two    stack_bowls_two_400ep"
  "stack_bowls_three  stack_bowls_three_500ep"
)

echo "==== 四任务顺序采集开始 $(date '+%F %T')(GPU ${GPU})===="
for entry in "${TASKS[@]}"; do
  read -r task cfg <<< "$entry"
  log="${LOG_DIR}/${task}_${cfg}.log"
  echo ""
  echo ">>>> [$(date '+%F %T')] 开始采集 ${task}(config: ${cfg}),日志: ${log}"

  if bash collect_data.sh "$task" "$cfg" "$GPU" 2>&1 | tee "$log"; then
    done_n=$(ls "external/robotwin_local/data/${task}/${cfg}/data" 2>/dev/null | wc -l)
    echo ">>>> [$(date '+%F %T')] ${task} 完成,已有 ${done_n} 个 episode"
  else
    echo "!!!! [$(date '+%F %T')] ${task} 采集异常退出(exit=$?),中止后续任务。"
    echo "!!!! 排查日志 ${log} 后重跑本脚本即可断点续采。"
    exit 1
  fi
done

echo ""
echo "==== 全部四个任务采集完成 $(date '+%F %T')===="
echo "下一步(转 ACT 训练格式,纯 CPU,很快):"
echo "  cd external/robotwin_local/policy/ACT"
echo "  bash process_data.sh adjust_bottle      adjust_bottle_200ep      200"
echo "  bash process_data.sh grab_roller        grab_roller_300ep        300"
echo "  bash process_data.sh stack_bowls_two    stack_bowls_two_400ep    400"
echo "  bash process_data.sh stack_bowls_three  stack_bowls_three_500ep  500"
