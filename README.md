# mujoco_t800_nav

EngineAI **T800 人形机器人**的 **运控RL（强化学习运动控制）** + **VLN（视觉语言导航）** 项目，基于 [MuJoCo](https://mujoco.org/) 仿真。

> 项目整体目标是「运控RL + VLN 导航」：先用强化学习让 T800 学会站立、深蹲站起、地面站起等运动技能，再叠加视觉语言导航（VLN）能力。当前仓库已完成运控RL 部分，VLN 部分尚在规划中。

## 机器人

- 型号：EngineAI **T800**（深圳中青机器人 / EngineAI）
- 关节数：**25** 个自由度 = 双腿 12（髋/膝/踝 pitch·roll·yaw）×2 + 腰 yaw 1 + 双臂 10（肩/肘 pitch·roll·yaw）×2 + 头 2（pitch·yaw）
- 模型来源：`models/t800/`（EngineAI 官方 MuJoCo 模型，**BSD 3-Clause** 许可）

## 目录结构

```
mujoco_t800_nav/
├── t800_common.py           # 共享工具：模型加载、关节寻址、PD 控制、观测/奖励物理量
├── 01_load_t800_viewer.py   # 加载模型并用 viewer 观察
├── 02_control_interface.py  # 状态/传感器/力矩下发接口
├── 03_pd_stand.py           # 手写 PD 站立（非 RL 基线）
├── 04_stand_env.py          # 站立 Gymnasium 环境
├── 05_ppo.py                # 手写 PPO（PyTorch，不用 stable-baselines3）
├── 06_eval_stand.py         # 评估站立策略
├── 07_get_up.py             # 「深蹲→站立」环境
├── 08_train_getup.py        # 训练深蹲站起策略
├── 09_eval_getup.py         # 评估深蹲站起策略
├── 10_get_up_ground.py      # 「地面折叠蹲→深蹲交接」环境
├── 11_train_getup_ground.py # 训练地面站起策略
├── 12_eval_getup_ground.py  # 评估地面站起策略
└── models/t800/             # T800 机器人 MuJoCo 模型（URDF/XML/mesh/纹理）
```

## 环境安装

- Python 3.12，conda 环境名 `t800`
- MuJoCo **4.0**
- 依赖：`gymnasium`、`numpy`、`torch`

```bash
conda create -n t800 python=3.12 -y
conda activate t800
pip install mujoco gymnasium numpy torch
```

## 快速开始

```bash
conda activate t800

# 1. 加载模型并用 viewer 观察
python 01_load_t800_viewer.py

# 2. 手写 PD 站立（非 RL 基线，约 1~2s 后会前倾，属预期）
python 03_pd_stand.py --headless

# 3. 站立环境自检（含吞吐量实测）
python 04_stand_env.py

# 4. 训练站立策略（默认 CPU，2000 迭代）
python 05_ppo.py --train 2000

# 5. 评估站立策略
python 06_eval_stand.py --viewer
```

## 训练流水线

项目按「由易到难」分三大任务阶段，每个阶段都是 **环境 → 训练 → 评估** 的完整闭环：

### 阶段一：站立平衡（01–06）

从加载模型、打通控制接口，到手写 PPO 学会双腿站立（含抗扰动）。

| 文件 | 作用 | 运行 |
| --- | --- | --- |
| `01_load_t800_viewer.py` | 加载模型 + viewer 观察 | `python 01_load_t800_viewer.py` |
| `02_control_interface.py` | 验证状态/传感器/力矩三条链路 | `python 02_control_interface.py` |
| `03_pd_stand.py` | 手写 PD 站立（非 RL 基线） | `python 03_pd_stand.py [--headless]` |
| `04_stand_env.py` | 站立 Gymnasium 环境（动作 4 维 / 观测 39 维） | `python 04_stand_env.py` |
| `05_ppo.py` | 手写 PPO 训练 | `python 05_ppo.py --train 2000` |
| `06_eval_stand.py` | 评估站立 / 抗扰动 | `python 06_eval_stand.py [--viewer] [--disturb]` |

### 阶段二：深蹲 → 站立（07–09）

从随机深蹲姿态站起并站住，采用四档课程学习（浅蹲 → 深蹲）。

| 文件 | 作用 | 运行 |
| --- | --- | --- |
| `07_get_up.py` | 深蹲→站立环境（动作 8 维 / 观测 45 维） | `python 07_get_up.py` |
| `08_train_getup.py` | 训练（默认 2000 迭代 + 课程） | `python 08_train_getup.py --iters 2000` |
| `09_eval_getup.py` | 评估站起成功率 | `python 09_eval_getup.py --ckpt checkpoints_getup/ppo_getup_final.pt` |

### 阶段三：地面折叠蹲 → 深蹲交接（10–12）

从地面折叠蹲（tuck，最深可达地面姿态）站起到深蹲交接带，需腿 + 臂协同。

| 文件 | 作用 | 运行 |
| --- | --- | --- |
| `10_get_up_ground.py` | 地面站起环境（动作 12 维 / 观测 62 维） | `python 10_get_up_ground.py` |
| `11_train_getup_ground.py` | 训练（含课程与预登记判据） | `python 11_train_getup_ground.py --iters 2000 [--course auto]` |
| `12_eval_getup_ground.py` | 评估交接成功率 | `python 12_eval_getup_ground.py --ckpt checkpoints_getup_ground/ppo_gg_final.pt` |

## 关键技术点

- **手写 PPO**（`05_ppo.py`）：不依赖 stable-baselines3，自己实现共享主干的 Actor-Critic（高斯策略 + 价值网络）、GAE 优势估计、clip 目标与多 epoch 更新。
- **PD 内环 + RL 目标角**：动作输出目标角偏移（站立）或绝对目标角（站起），PD 控制器换算为力矩，内环以 1kHz 物理步运行、策略以 100Hz 决策（`frame_skip=5`）。
- **观测/动作维度**：
  - 站立：观测 39 维，动作 4 维（仅双踝 pitch/roll —— 站立平衡的可验证子空间）
  - 深蹲站起：观测 45 维，动作 8 维（髋/膝/踝 pitch + 踝 roll ×2）
  - 地面站起：观测 62 维，动作 12 维（8 腿 + 4 臂）
- **奖励与课程设计**：站立用直立度 + 高度 + 足底接触 + 存活等项；站起用「高度增量/记录势」进步奖励 + 成功带收割 + 四档深蹲课程，并用 `value_loss` 尖峰、成功带累计步数等预登记判据防止 reward hacking。
- **向量化**：训练内环用 `t800_common.build_addr` 预计算地址 + NumPy 花式索引，避免每物理子步做上百次字符串寻址。

## 注意事项

- **模型路径必须用绝对路径**：MuJoCo 4.0 下相对路径加载会非确定性失败，`t800_common.py` 已统一用绝对路径。
- **passive viewer 退出段错误**：MuJoCo 4.0 passive viewer 退出时 GLFW 被双重 `terminate`，脚本结尾用 `time.sleep(0.5); os._exit(0)` 跳过 `atexit` 里多余的 `glfw.terminate`。
- **CPU vs GPU**：本任务是「单环境 + 小网络」，实测 CPU 比 GPU 快；只有改成多环境并行 + 大批量时 GPU 才划算（可用 `--gpu` 切换）。
- 检查点目录：站立 `checkpoints/`、深蹲站起 `checkpoints_getup/`、地面站起 `checkpoints_getup_ground/`。

## 许可

- 机器人模型（`models/t800/`）：EngineAI，BSD 3-Clause。
- 其余代码：待定。
