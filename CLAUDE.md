# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库定位

TronCamp 比赛选手包：在 RoboTwin 2.0 仿真里用 Tron2 双臂机器人完成四个难度递增的操作任务（T1 `adjust_bottle` → T2 `grab_roller` → T3 `stack_bowls_two` → T4 `stack_bowls_three`，只有 T4 计分），训练 ACT（Action Chunking Transformer，~80M，单张 24G 显卡可训）。四题共用同一条「采集 → 数据转换 → 训练 → 评测 → 提交」流水线。

运行环境为独立 conda 环境 `troncamp_env`（Python 3.10，PyTorch + SAPIEN + cuRobo 0.8.0）。装完用 `python setup/env_check.py` 自检。

## 常用命令

流水线三步用同一组参数 `<task_name> <task_config> <num>` 串起来，产出数据键 `sim-<task>-<config>-<num>`：

```bash
# 1. 自采数据（必须走此 wrapper，它先打 Tron2 自碰撞 runtime patch；裸跑官方 collect_data.py 专家成功率为 0）
bash collect_data.sh adjust_bottle adjust_bottle_200ep 0        # 参数: <task_name> <task_config> [gpu_id]

# 2. 转 ACT 16-D 训练格式
( cd external/robotwin_local/policy/ACT && bash process_data.sh adjust_bottle adjust_bottle_200ep 200 )

# 3. 训练（turnkey 超参写死在 train.sh: kl_weight 10 / chunk 50 / hidden 512 / lr 1e-5 / state_dim 16）
( cd external/robotwin_local/policy/ACT && bash train.sh adjust_bottle adjust_bottle_200ep 200 0 0 )
# 产出 → external/robotwin_local/policy/ACT/act_ckpt/act-<task>/<config>-<num>/（policy_best.ckpt / policy_last.ckpt + dataset_stats.pkl）

# 4. 公开 seed 本地自评（与官方评测同一内核，同分布不同种子；评的是 ckpt-dir 里的 policy_last.ckpt）
python starter/eval_local.py --track T1 --ckpt-dir external/robotwin_local/policy/ACT/act_ckpt/act-adjust_bottle/adjust_bottle_200ep-200

# 单 seed 单条 rollout 数值核查（末态 sr/graded，不渲染视频）
python starter/watch_rollout.py --track T1 --ckpt-dir <ckpt_dir> --seed 0

# 5. 提交（唯一通道；T1 只交权重，T2/T3/T4 加 --code-dir external/robotwin_local）
python submit/submit.py --token-file <token文件> --track T1 --ckpt <policy_best.ckpt 路径>
```

### 测试

评测内核有纯 Python 单测（无 sim/GPU 依赖）：

```bash
pytest recipes/eval/tests/                          # 全部
pytest recipes/eval/tests/test_graded_score.py      # 单个文件
pytest recipes/eval/tests/test_graded_score.py::test_full_success_is_one   # 单个用例
```

## 架构

### 顶层布局

- `collect_data.sh` → `recipes/rollout/collect_tron2_data.py`：打 Tron2 runtime patch 后调 RoboTwin 官方采集，只留成功 episode。
- `recipes/eval/`：**官方评测内核**（选手本地与官方评测机跑同一套代码）。
- `recipes/train/`、`recipes/eval/act_eval.sh`：早期 phase 脚本（`act_train.sh`/`act_process.sh` 等），是 `policy/ACT` 下 turnkey 脚本的带环境变量可调版本。
- `starter/`：选手入口包装（`eval_local.py`、`watch_rollout.py`、公开 100 seed `public_seeds.json`）。
- `submit/submit.py`：唯一提交通道（仅标准库，multipart 上传 ckpt + 可选代码包）。
- `external/robotwin_local/`：内嵌的 RoboTwin 2.0（含 Tron2 embodiment 接入、四任务调好专家、task config、`policy/ACT` 训练栈、`envs/curobo` 运动规划）。
- `embodiments/tron2_v5_DACH_validing/`：Tron2 机器人 URDF / mesh / curobo 碰撞配置。

### 评测内核（recipes/eval）

分层设计，核心是 **result 契约** `{sr, n_repeats, n_episodes, per_repeat, track, graded}`——sim/GPU 侧生产、榜单后端只消费：

- `act_contract.py`：纯编排核心（无 GPU），对「seed 表 × repeats」循环注入的 rollout backend；`TASK_BY_TRACK` 是赛道→任务的唯一映射源。
- `run_act_eval.py`：真实 backend，in-process 单 GPU 跑 ACT rollout（复用 RoboTwin `eval_policy` 的模型/配置机制），强制 `policy_name=ACT`。
- `graded_score.py`：T4 分级计分（按末态叠到第几层给分，三层各 1/3）。**结构性约束：`graded_stack_score` 只能由编排层在每 episode 末态调用一次，绝不能接进 env 的逐步 `check_success`**（历史 4M-average bug，测试有钉）。
- T1–T3 契约里不含 `graded` 键，纯 SR 过阈值解锁。

`starter/eval_local.py` 只是 subprocess 包装：定位内核 + 公开 seed 表 + `--robotwin-root`。

### 采集链的关键点

- 专家 = `external/robotwin_local/envs/<task>.py` 的 `play_once()`；改进专家直接改该文件。
- `recipes/rollout/tron2_runtime_patch.py` 在运行时 monkeypatch RoboTwin（禁自碰撞、修 grasp 选择/终点漂移等 Tron2 集成问题），**刻意不改 external 里的 RoboTwin 源码**。
- 默认单趟采集（`ROBOTWIN_SAVE_DURING_SEARCH=1`）：两阶段「搜索→回放」会因 curobo 构造重置 numpy 全局 RNG 导致场景不一致而崩溃。
- 采集需要 `ffmpeg` 可执行，缺失时成功集静默保存失败。

### Task config（external/robotwin_local/task_config/）

- `<task>_200ep.yml`：采集 config（集数/域随机化/场景）。T1 已给 turnkey 的 `adjust_bottle_200ep.yml`；T2–T4 拷贝改名自建。
- `<task>_clean.yml`：评测 config，随包下发，**勿改**。
- `policy/ACT/deploy_policy.yml`：推理架构配置（hidden 512 / chunk 50），**必须与 ckpt 训练时架构一致**，不匹配会在加载时报 state_dict size mismatch（768/chunk-100 的大模型用 `deploy_big.yml`）。

### 硬性约束

- `state_dim = 16`（Tron2 双臂 7+1 DOF × 2），全链一致。
- curobo 必须是内嵌的 **0.8.0**（`env_check.py` 断言，拒绝 0.7.x 残留）。
- 安装后须还原 `__KIT_ROOT__` 占位（*.yml 里的路径占位符），漏还原 curobo 规划器起不来；`env_check.py` 会扫出残留。
- RoboTwin 运行时路径可用环境变量 `TRON2_ROBOTWIN_DIR` 覆盖，默认 `external/robotwin_local`。

## 文档

完整分步（安装/采集/训练/评测/提交）见参赛文档：https://limx-troncamp.github.io/troncamp-web-mani/doc.html
