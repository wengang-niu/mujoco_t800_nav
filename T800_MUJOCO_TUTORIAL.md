# T800 人形机器人 MuJoCo 实战教程

> 本教程**参考** Bilibili 视频教程 [BV1TLNn6HEsm](https://www.bilibili.com/video/BV1TLNn6HEsm/)（黑马程序员《零基础 AI 具身智能机器人实战全套课程》，21 集）的知识脉络重新编写，**不是视频字幕或逐字稿**。
>
> 两个保证：
> 1. **每个代码都能跑**——教程配套 `tutorial/` 目录下的脚本全部在本地实测通过；
> 2. **不依赖课程私有资料**——机器人模型与行走策略直接取自众擎官方 SDK（`engineai_robotics_native_sdk/`，已下载到本仓库），任何 Linux 机器都能复现。

---

## 环境准备

```bash
conda create -n t800 python=3.11 -y
conda activate t800
pip install mujoco==3.4 numpy==2.3.5 MNN
```

- `mujoco`：物理仿真引擎（3.4 或 4.0 均可，教程用 3.4 与视频一致）。
- `MNN`：阿里开源的轻量推理框架，用于加载官方 `.mnn` 行走策略（本教程第 11 课起需要）。
- viewer 相关课程（第 7/10/16 课）需要图形界面；无显示环境请用 `xvfb-run`。

**目录结构**：

```
mujoco_t800_nav/
├── engineai_robotics_native_sdk/   # 众擎官方 SDK（模型 + MNN 策略 + 参数）
└── tutorial/
    ├── common.py                   # 共享模块：SDK 参数、模型加载、观测/推理/PD
    ├── 02_hello_world.py           # MuJoCo 入门：方块自由落体
    ├── 03_load_t800.py             # 加载 T800
    ├── 04_joints_links.py          # 认识 joint 和 link
    ├── 05_coordinates.py           # 相对坐标与绝对坐标
    ├── 06_state.py                 # 状态参数 nq/nv/nu/qpos/qvel/ctrl
    ├── 07_loop.py                  # 仿真主循环
    ├── 08_pd.py                    # PD 控制
    ├── 09_torque_limit.py          # 力矩范围
    ├── 10_manual.py                # 手动控制（键盘）
    ├── 11_policy_infer.py          # AI 策略：MNN 推理
    ├── 12_observation.py           # 观测空间 1083 的来历
    ├── 13_action_to_motor.py       # 动作 → 目标角 → PD → 力矩
    ├── 14_debug.py                 # 调试策略：日志 + 启动过渡
    ├── 15_base_height.py           # 初始化高度：接触点扫描
    ├── 16_wasd.py                  # 二次开发：上下左右键遥控行走
    ├── 17_box.py                   # 在世界里构建木块
    ├── 18_kick_box.py              # 踢着木块往前走
    ├── 19_hold_object.py           # 手持物体走路（坐标绑定）
    └── 20_task_decompose.py        # 调试任务拆解（固定 base 只动右臂）
```

所有脚本从仓库根目录运行，例如 `python tutorial/02_hello_world.py`。

---

## 第 1 课 · 课程介绍

**目标**：理解「具身智能」与整门课的链路，知道每一步要解决什么问题。

人形机器人开发的完整链路是：

```
MuJoCo 仿真 → 加载模型 → 读懂状态 → 施加力矩（PD） → AI 策略推理 → 目标角映射 → 行走 → 与物体交互
```

- **MuJoCo**：物理仿真引擎，替你算好重力、碰撞、关节力矩，你不用手推牛顿力学。
- **T800**：众擎（EngineAI）的人形机器人，本教程用它的官方 MuJoCo 模型与强化学习行走策略。
- 整门课从「在 MuJoCo 里让一个方块掉下来」开始，一路做到「让 T800 拿着木块走路」。

后续每课对应视频的一集，但代码都是**可直接运行、并针对本机 Linux 环境调整过**的版本。

---

## 第 2 课 · MuJoCo 入门

**目标**：装好环境，用一段内联 XML 在 MuJoCo 里搭一个方块让它自由落体。

MuJoCo 的世界用 XML（MJCF 格式）描述：地面（plane）、方块（box）、重力（gravity）、时间步长（timestep）。加载模型得到 `model`（结构/参数），再创建 `data`（运行时状态），每调用一次 `mj_step` 就推进一个时间片。

代码：[tutorial/02_hello_world.py](tutorial/02_hello_world.py)

```bash
python tutorial/02_hello_world.py
```

输出（方块从 1 米高处落下，每 0.2 秒打印一次高度）：

```text
t=0.000s, z=1.000000m
t=0.200s, z=0.801838m
t=0.400s, z=0.211276m
t=0.600s, z=0.049658m
t=0.800s, z=0.049892m
...
```

方块约 0.44 秒落到地面，之后高度稳定在约 0.049m（box 半尺寸 0.05 的中心高度）。

---

## 第 3 课 · 加载 T800

**目标**：把众擎官方 SDK 里的 T800 模型加载进 MuJoCo，理解几个数量。

T800 模型文件是 `engineai_robotics_native_sdk/assets/resource/t800.xml`，它 include 了机器人和地面。加载后打印：

- `nbody = 31`：身体部件（含 world）。
- `njnt = 26`：关节（含 1 个浮基自由关节 + 25 个转动关节）。
- `nu = 25`：电机（执行器）数量。
- `nq = 32` / `nv = 31`：广义位置/速度维度。浮基贡献 7 个位置量（位置 3 + 四元数 4）但只有 6 个速度量，所以 `nq = nv + 1`。

代码：[tutorial/03_load_t800.py](tutorial/03_load_t800.py)

```bash
python tutorial/03_load_t800.py
```

关键点：25 个关节从 `J00_HIP_PITCH_L` 到 `J24_HEAD_YAW`，命名规整，后续用**名称**寻址而不是靠数组下标硬编码。

---

## 第 4 课 · 认识 T800 的 joint 和 link

**目标**：遍历所有关节（joint）和连杆（body），掌握按名称/ID 互相查找的 API。

- **link / body**：刚体（躯干、大腿、小腿、脚）。
- **joint**：连接两个 body 的关节，决定它们如何相对运动。
- 用 `mujoco.mj_id2name(model, mjtObj.mjOBJ_JOINT, id)` 得到关节名，用 `mj_name2id` 反查 ID。

代码：[tutorial/04_joints_links.py](tutorial/04_joints_links.py)

```bash
python tutorial/04_joints_links.py
```

关键点：关节 0 是浮基自由关节（`name=None`），它占据 qpos 前 7 位（位置 + 四元数）。所以「第几个关节」≠「qpos 的第几个元素」，必须用 `jnt_qposadr` / `jnt_dofadr` 寻址。

---

## 第 5 课 · 相对坐标与绝对坐标

**目标**：区分局部坐标 `model.body_pos` 与世界坐标 `data.xpos`。

- `model.body_pos[bid]`：body 相对**父 body** 的模型定义位置（XML 里写的 `pos`）。
- `data.xpos[bid]`：经过正向运动学（`mj_forward`）后，body 在世界坐标系里的位置。

代码：[tutorial/05_coordinates.py](tutorial/05_coordinates.py)

```bash
python tutorial/05_coordinates.py
```

示例输出（注意脚的局部坐标是 0，而世界 Z 是 0.077m）：

```text
LINK_KNEE_PITCH_L    [-0.0377, 0.0, -0.2879]     [-0.0236, 0.1195, 0.554]
LINK_FOOT_L          [ 0.0, 0.0, 0.0]            [-0.0252, 0.1195, 0.0774]
```

把 body 局部点转世界系：`world = data.xpos[bid] + R @ local_point`，其中 `R` 是基座旋转矩阵。

---

## 第 6 课 · 状态参数

**目标**：理解 `qpos` / `qvel` / `ctrl` 三个核心数组。

| 数组 | 含义 | 长度 |
|---|---|---|
| `qpos` | 广义位置（浮基位姿 + 各关节角） | nq=32 |
| `qvel` | 广义速度（浮基线/角速度 + 关节角速度） | nv=31 |
| `ctrl` | 发给执行器的控制量（本模型即关节力矩 N·m） | nu=25 |

代码：[tutorial/06_state.py](tutorial/06_state.py)

```bash
python tutorial/06_state.py
```

机器人控制本质就是：**读 `qpos`/`qvel` 了解当前状态 → 改 `ctrl` 施加控制 → `mj_step` 推进一步 → 循环**。

---

## 第 7 课 · 仿真主循环

**目标**：跑通「读取状态 → 写控制 → `mj_step`」的主循环，观察无控制时机器人因重力倒地。

`mujoco.mj_step(model, data)` 是推进仿真「一帧」的核心调用。物理步长 0.002s（模型 XML 里 `timestep`）。

代码：[tutorial/07_loop.py](tutorial/07_loop.py)

```bash
python tutorial/07_loop.py             # 无窗口，打印高度/倾角
python tutorial/07_loop.py --viewer    # 打开 viewer 观看倒地过程
```

输出显示：所有电机力矩为 0 时，机器人 base_z 从 1.02 一路跌到 0.48，俯仰角转到 −110°——浮基双足没有力矩必然倒下。

---

## 第 8 课 · PD 控制

**目标**：用 PD 控制器给关节「装上肌肉」，理解为什么纯固定 PD 仍站不稳。

PD 把关节拉到目标角：

```
tau = kp * (q_des - q) + kd * (0 - qd)
```

`kp` 是刚度（弹簧），`kd` 是阻尼（刹车）。本教程用官方 SDK 的 25 个关节 kp/kd。

代码：[tutorial/08_pd.py](tutorial/08_pd.py)

```bash
python tutorial/08_pd.py
```

输出：PD 把关节角误差压在 0.1~0.6 内，但基座俯仰角在约 1.3 秒后一路转到 89°——**PD 只能让关节跟踪目标，浮基双足要站稳必须靠策略实时调整目标角**（这正是后续引入 AI 策略的原因）。

---

## 第 9 课 · 力矩范围

**目标**：读懂每个电机的力矩上限，用 `np.clip` 把超限力矩裁到安全范围。

T800 的 25 个电机力矩上限（`actuator_ctrlrange`）各不相同：髋/膝俯仰 ±415 N·m，踝/肩/肘 ±160，肘偏航/头 ±52。真实硬件有物理极限，算出的力矩必须裁剪。

代码：[tutorial/09_torque_limit.py](tutorial/09_torque_limit.py)

```bash
python tutorial/09_torque_limit.py
```

`np.clip(tau, -limit, limit)` 把超限值压到上下限——这是**保证真实硬件不烧坏**的工程前提。

---

## 第 10 课 · 手动控制

**目标**：用键盘实时改某个关节的目标角，经 PD 跟踪，打通「手动 → 单关节」链路。

代码：[tutorial/10_manual.py](tutorial/10_manual.py)（用 MuJoCo viewer 的按键回调，跨平台）

```bash
python tutorial/10_manual.py
```

按 `W`/`S` 增大/减小目标角，`Q` 退出。观察：改的是**目标角**，实际运动由 PD 产生力矩驱动。

> 为什么要引入 AI：T800 有 25 个关节，手动逐关节调既「维度爆炸」（人脑难以协调 20+ 维）又「关节耦合」（互相影响），所以要用策略模型自动输出 22 个关节的动作。

---

## 第 11 课 · AI 策略生成关节运动指令

**目标**：加载官方 MNN 行走策略，完成一次「1083 维观测 → 22 维动作」的推理。

官方策略文件是 `engineai_robotics_native_sdk/assets/config/t800/rl_walking_example/policy/*.mnn`。用 `MNN` 包加载后：输入 `(1,1083)`，输出 `(1,22)`。

代码：[tutorial/11_policy_infer.py](tutorial/11_policy_infer.py)

```bash
python tutorial/11_policy_infer.py
```

```text
观测维度 = 1083（= 15×72 + 3 = 1083）
动作维度 = 22（22 个受控关节）
前 6 个动作： [-0.0712 -0.3471 -0.1378  0.1378  0.0706  0.418 ]
```

策略网络输入 1083 个数据、输出 22 个动作——这就是机器人的「大脑」。受控的是 22 个关节（双腿 12 + 双臂 10），腰和头 2 个关节不参与策略。

---

## 第 12 课 · 观测空间（1083 的来历）

**目标**：拆解 1083 = 15 帧历史 × 72 维单帧 + 3 维指令，理解每段含义。

单帧 **72 维**观测的顺序（必须与训练一致）：

| 段 | 维度 | 含义 | 缩放 |
|---|---|---|---|
| 关节位置偏差 | 22 | q − default_q | ×1.0 |
| 关节速度 | 22 | qd | ×0.05 |
| 上一动作 | 22 | prev_action | ×1.0 |
| 身体角速度 | 3 | IMU 陀螺（body 系） | ×1.0 |
| 投影重力 | 3 | 直立≈[0,0,−1] | ×1.0 |

单帧只有 72 维还不够判断「要摔倒还是正恢复平衡」，所以要缓存过去 **15 帧**历史（15×72=1080），末尾再拼 **3 维指令**（前后速度、左右速度、原地转向角速度），得 1083。

代码：[tutorial/12_observation.py](tutorial/12_observation.py)

```bash
python tutorial/12_observation.py
```

---

## 第 13 课 · AI 指令到机器人肌肉（动作映射）

**目标**：理解网络输出的 22 维动作如何变成力矩。

网络输出是归一化动作，**不能直接当力矩**。正确链路：

```
action(22) → 裁剪 → q_des = default_q + action × action_scale → PD → tau → 裁剪 → ctrl
```

- `default_q`：默认站立姿态（基准）。
- `action_scale`：把动作映射成目标角偏移（每个关节不同，见 `common.py`）。
- 以默认姿态为基准只学「动多少」，称为**残差控制**，比直接学绝对角度更容易训练。

代码：[tutorial/13_action_to_motor.py](tutorial/13_action_to_motor.py)

```bash
python tutorial/13_action_to_motor.py
```

---

## 第 14 课 · 仿真调试策略

**目标**：把整条链路跑起来，用日志观察数值是否发散，并掌握「启动过渡」。

完整闭环 = 观测 → 推理 → 动作 → 目标角 → PD → `mj_step`。本课演示两个关键调试点：

1. **日志为王**：打印基座高度、俯仰角、动作量级，数值发散一眼可见。
2. **启动过渡**：前 0.5 秒把 kp/kd 从 20% 渐变到 100%，避免初始时刻阶跃冲击。

代码：[tutorial/14_debug.py](tutorial/14_debug.py)

```bash
python tutorial/14_debug.py --seconds 10
```

```text
  t(s)   base_x   base_z pitch_deg  max|action|
  0.00   -0.000    1.019      0.01       0.807
  5.00    0.465    1.015     -0.13       3.213
 10.00    1.04     1.017      0.04       4.87
结束：前进 1.04m，基座高度 1.017m（稳定）
```

机器人稳定向前行走，基座高度恒定、俯仰角接近 0——这就是「策略接管成功」的标志。

---

## 第 15 课 · 初始化高度设置

**目标**：用接触点扫描找出「脚底刚好触地」的基座高度。

初始高度过高机器人会先自由落体（落地冲击），过低则脚陷进地面（被接触求解弹起）。用代码在 0.50~1.50m 区间按 0.02m 步长扫描，统计与地面的接触点数。

代码：[tutorial/15_base_height.py](tutorial/15_base_height.py)

```bash
python tutorial/15_base_height.py
```

输出显示接触数从 34（深埋）逐步降到 0，**在 z=1.02 时刚变为 0**——这就是合适的初始化高度（`common.py` 里的 `BASE_Z`）。

---

## 第 16 课 · 运动控制二次开发（上下左右键遥控）

**目标**：在行走闭环上接键盘，实时改写三维指令，实现前进/后退/转向。

策略的「运动命令」就是 1083 维观测末尾那 3 维指令 `[vx, vy, yaw_rate]`。改它就能改机器人行进方向，而不用动策略网络本身。

代码：[tutorial/16_wasd.py](tutorial/16_wasd.py)

```bash
python tutorial/16_wasd.py
```

按键：`↑/↓` 前进后退，`←/→` 左右转，`SPACE` 停止。

---

## 第 17 课 · 在世界里构建木块

**目标**：在 MuJoCo 世界里加一个带自由关节的红色木块。

用 `mujoco.MjSpec` 往模型里追加一个 body（`freejoint` + box `geom`），再编译回模型。

代码：[tutorial/17_box.py](tutorial/17_box.py)

```bash
python tutorial/17_box.py
```

```text
新增木块后 body 数量 = 32
木块落点 xyz = [ 1.5 -0.   0.1]
```

---

## 第 18 课 · T800 踢着木块往前走

**目标**：在机器人前方放木块，跑行走策略，观察机器人把它推着/踢着向前。

代码：[tutorial/18_kick_box.py](tutorial/18_kick_box.py)

```bash
python tutorial/18_kick_box.py
```

```text
  t(s)   robot_x    box_x
  0.00   -0.000    1.200
  8.00    1.04     1.30
结束：机器人 x=1.04m，木块 x=1.30m（被推动前进）
```

---

## 第 19 课 · 手持物体走路（坐标绑定）

**目标**：用「坐标绑定」让木块跟随右手腕，演示手持物体走路。

真实抓取要考虑摩擦、形变、接触点，太复杂。入门用**坐标绑定**简化：每个物理步把木块世界坐标设为右手腕世界坐标。关键技巧是把木块设为 `contype=0 / conaffinity=0`（不参与碰撞），否则木块会与手部碰撞干扰行走。

代码：[tutorial/19_hold_object.py](tutorial/19_hold_object.py)

```bash
python tutorial/19_hold_object.py
```

```text
  t(s)  wrist_x    box_x
  0.00    0.083    0.300
  7.00    0.988    0.985
结束：木块已随右手腕一起前进（坐标绑定生效）
```

---

## 第 20 课 · 调试任务拆解（最小原子任务）

**目标**：把「边走路边抓取」这类复合需求拆成互不干扰的子任务，先单点验证。

复合需求一出问题很难定位（是腿？是手臂重心？还是木块？）。做法是**任务拆解**：把走路关掉、把重心锁住，先单独验证「抬右臂」这一块。

代码：[tutorial/20_task_decompose.py](tutorial/20_task_decompose.py)

```bash
python tutorial/20_task_decompose.py
```

做法：固定 base（每步把浮基位姿/速度钉回初始值），只让右臂在两个目标姿态间用 `alpha` 插值 + `smoothstep` 平滑过渡。

```text
t=0.00s  右肘角 J21=-0.250 rad
t=0.80s  右肘角 J21=-0.714 rad   （抬起）
t=2.40s  右肘角 J21=-0.233 rad   （放下）
结束：固定 base、只动右臂的最小原子任务跑通。
```

---

## 第 21 课 · 总结

学完这套教程，你走通了人形机器人控制的完整链路：

```
MuJoCo 仿真 → 加载模型(3) → joint/link 与坐标(4,5) → 状态(6) → 主循环(7)
           → PD(8) → 力矩限制(9) → 手动(10) → AI 策略(11) → 观测(12)
           → 动作映射(13) → 调试(14) → 初始化(15) → 遥控行走(16)
           → 木块(17) → 踢木块(18) → 持物(19) → 任务拆解(20)
```

三个进阶方向：MJCF/URDF 仿真参数对齐、MPC/强化学习步态、复杂任务规划（感知 + SLAM）。

---

## 附录 A · 关键参数速查（官方 SDK）

来源：`engineai_robotics_native_sdk/assets/config/t800/rl_walking_example/default.yaml`（完整定义见 [tutorial/common.py](tutorial/common.py)）。

| 参数 | 值 | 说明 |
|---|---|---|
| 观测单帧 / 历史 / 指令 | 72 / 15 / 3 | 1083 = 72×15 + 3 |
| 受控关节 | 22（腿 12 + 臂 10） | 腰 J12、头 J23/J24 不参与策略 |
| 动作缩放 action_scale | 腿 [0.5,0.2,0.2,0.5,0.5,0.2]，臂 [0.2,0.2,0.05,0.2,0.05] | 各关节不同 |
| PD 增益 kp/kd | 腿 180/5、踝 40/0.3、臂 60/0.3 … | 见 common.py |
| 默认姿态 default_q | 髋 −0.06、膝 0.12、踝 −0.06 … | 行走基准姿态 |
| 控制周期 / 物理步长 | 0.01s / 0.002s | 每策略步 = 5 个物理子步 |
| 指令缩放 / 裁剪 | [2,2,1] / obs±100 / action±100 | 观测、动作都裁剪 |

## 附录 B · 常见问题

- **`ModuleNotFoundError: No module named 'common'`**：请从仓库根目录运行 `python tutorial/XX.py`，脚本目录会自动加入 `sys.path`。
- **`mujoco.viewer` 不存在**：脚本已显式 `import mujoco.viewer`；若你改代码，记得同样处理。
- **模型加载失败**：确认 `engineai_robotics_native_sdk/` 在仓库根目录；`common.py` 用绝对路径加载 `assets/resource/t800.xml`。
- **策略不走路 / 前空翻**：多半是观测顺序或缩放错（对照第 12/13 课），或初始化高度不对（第 15 课），或启动过渡缺失（第 14 课）。
- **无显示环境**：viewer 课（7/10/16）用 `xvfb-run -a python tutorial/XX.py`。
