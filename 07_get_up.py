"""07 — T800「深蹲→站立」Gymnasium 环境（坐姿站起可行性已验证的第一档）。

可行性研究结论（2026-11-13，全部经实测复跑确认）：
- 足平贴地深蹲族可稳持到膝屈 k≈1.35（髋 0.80m）；k=1.20（髋 0.84m）已被开环
  斜坡实测 1~3s 站起（膝峰值 119 N·m = 限额 57%、踝从未触 35 N·m、CoP 全程 ±2cm）。
- k≥1.6 静态膝力矩需求 214 N·m 超过 0.5×ctrlrange=207.5 N·m，物理不可行；
  地面级折叠坐（大腿/手腕支撑、骨盆底 0.128m）腿力矩站不起，需手臂/动量，留 v2。
- 深蹲 PD 必须 Kp=800/Kd=40（Kp=400 会被踝负载刚度 ~350 N·m/rad 压塌、缓慢前倾）。

与 04_stand_env 的关键差异：
- 动作是「绝对目标角」而非小偏移：q_des = clip(a × 半行程, 限位±0.03)。
  坐姿髋 −0.625 rad 超出任何小偏移尺度，只能用绝对目标表达；a=0 即站立目标。
  目标经变化率低通（1.2 rad/s）执行，避免阶跃轰击。
- 8 维 = 双髋 pitch、双膝 pitch、双踝 pitch、双踝 roll；髋 roll/yaw、腰、臂、头锁 0。
- 奖励（v4d，2026-11-14 经 4 路评审 workflow 裁定，v4c 实训练后升级）：
  progress = 5×Δh 纯高度增量（马尔可夫——v4 实测 value_loss 尖峰 41→0 次、熵 9.6→7.6
  平台）；upright/feet 收割只在成功带内给（P1 防 hacking）；带外弱 upright 指引 0.02
  （h>0.75，升程姿态整形）。v4c 实测（L2 4%/L3 60%）证明 0.1/0.05 带内惩罚只是「固定税」
  （命中收割 0.73/步 的 1/10，带内 |a|₂ 均值 1.67-2.20 的 crank 吸引子纹丝不动，成功=
  骑线过杆：反事实关掉成功终局后 11/11「本应成功」回合过杆后 4-87 步全部崩溃），v4d 加码：
  (1) 带内 −0.4·mean(|a|) − 0.2·足力不对称 − 0.3·mean(q²@8 动作关节)：crank 态付收割的
  47-63%，crank-vs-quiet 梯度差 ≈48-69/回合；带边最坏角落净额 ≈0，不给「带边弃守」留口子；
  (2) 成功计数改为回合内累计（出带不清零）——旧版出带清零使 150 步封顶只约束连续段，
  flicker 出带再入循环可收割 ≈1350-1400/回合（10× 诚实站起，防黑评审实测漏洞）。
  成功带（EMA 高度≥0.95 且倾角≤0.22 且双足>100N 累计 150 步 → +20 提前终局）+ 柔和惩罚。
- 终止：NaN / 出界 / 关节超限 / stuck（500 步高度 EMA 无进步）/ 成功 / 2000 步超时。
  绝不因倒地终止——塌到地面允许自己再起。
- reset：从预解 IK 表采样深蹲姿态（足平贴地、已落稳 0.2s），yaw 全随机 + 小噪声；
  课程按膝屈深度分 4 档（set_level 控制）。

运行自检：python 07_get_up.py
"""
from __future__ import annotations

import time

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import t800_common as tc

# 动作关节（腿 12 维下标）：髋pitch L、膝 L、踝pitch L、踝roll L、髋pitch R、膝 R、踝pitch R、踝roll R
ACTION_LEG_IDX = np.array([0, 3, 4, 5, 6, 9, 10, 11], dtype=int)
# 各动作关节半行程（rad）——绝对目标角的缩放系数
ACTION_HALF_RANGE = np.array([2.79, 1.31, 0.68, 0.26, 2.79, 1.31, 0.68, 0.26])

KP_LEG, KD_LEG = 800.0, 40.0   # 深蹲所需 PD 刚度（可行性实验实测 Kp=400 压塌）
MAX_STEPS = 2000               # 每回合最大策略步数（20s @ 100Hz）
SUCCESS_STEPS = 150            # 成功带累计 150 步（1.5s，回合内计数、出带不清零）→ 提前终局
                               # （v4d 防黑修复：连续计数会被出带 flicker 重置、封顶失效）
SUCCESS_BONUS = 20.0           # 成功终局一次性奖励
STUCK_STEPS = 500              # 高度 EMA 无进步 500 步 → truncated
QDES_MAX_RATE = 0.006          # 目标角低通的最大关节速率（rad/步 = 0.6 rad/s @100Hz）。
                               # 实测结论：统一速率限幅会让行程短的踝提前归零、CoP 前移翻倒；
                               # 必须按剩余行程等比例扫向目标（= 所有关节同步到站）。
                               # 1~3s 斜坡 5/5 成功，0.6 rad/s ≈ T=1.04s，在验证过的成功区间内
SUCCESS_H = 0.95               # 成功带：EMA 高度阈值
SUCCESS_TILT = 0.22            # 成功带：倾角阈值（rad ≈ 12.6°）。实测深蹲站起后策略停在
                               # tilt 10~25° 缓漂（亚稳态），8.6° 过严导致进不了成功带
SUCCESS_FZ = 100.0             # 成功带：单足接触力阈值（N）
TARGET_HEIGHT = tc.TARGET_HEIGHT

# 课程 4 档：reset 采样膝屈 k 的区间（k 越大蹲越深）
CURRICULUM_K = [
    (0.10, 0.45),   # L0 浅蹲（髋 0.99~0.94m）
    (0.45, 0.85),   # L1 中蹲（0.94~0.87m）
    (0.85, 1.20),   # L2 深蹲（0.87~0.84m）
    (0.10, 1.20),   # L3 全混
]

_IK_CACHE: dict[float, tuple[float, float, float, float]] = {}


def _ik_solve(model: mujoco.MjModel, data: mujoco.MjData, k: float) -> tuple[float, float, float, float]:
    """解「足平贴地、踝对正足原点」的对称深蹲姿态（阻尼牛顿，实测可用）。

    返回 (hip_pitch, ankle_pitch, ankle_roll, base_z)，对称复制到双腿。
    残差：踝 x 对正足原点 / 足俯仰为 0 / 足侧倾为 0 / 足底 z=0（足高 0.064m）。
    """
    bid_ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_ANKLE_PITCH_L")
    bid_foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_FOOT_L")

    def residuals(h: float, a: float, r: float, bz: float) -> np.ndarray:
        mujoco.mj_resetData(model, data)
        data.qpos[0:7] = [0.0, 0.0, bz, 1.0, 0.0, 0.0, 0.0]
        for n in tc.JOINT_NAMES:
            data.qpos[tc.qpos_adr(model, n)] = 0.0
        for n, v in (("J00_HIP_PITCH_L", h), ("J06_HIP_PITCH_R", h),
                     ("J03_KNEE_PITCH_L", k), ("J09_KNEE_PITCH_R", k),
                     ("J04_ANKLE_PITCH_L", a), ("J10_ANKLE_PITCH_R", a),
                     ("J05_ANKLE_ROLL_L", r), ("J11_ANKLE_ROLL_R", -r)):
            data.qpos[tc.qpos_adr(model, n)] = v
        mujoco.mj_forward(model, data)
        an = data.xpos[bid_ankle]
        fo = data.xpos[bid_foot]
        xm = data.xmat[bid_foot].reshape(3, 3)
        return np.array([an[0], xm[2, 0], xm[1, 2], fo[2] - 0.064])

    h, a, r, bz = -1.0, -0.5, 0.0, 0.7
    for _ in range(80):
        f = residuals(h, a, r, bz)
        for idx, var in enumerate(("h", "a", "r", "bz")):
            f0 = f[idx]
            eps = 1e-3
            if var == "h":
                f1 = residuals(h + eps, a, r, bz)[idx]
            elif var == "a":
                f1 = residuals(h, a + eps, r, bz)[idx]
            elif var == "r":
                f1 = residuals(h, a, r + eps, bz)[idx]
            else:
                f1 = residuals(h, a, r, bz + eps)[idx]
            jac = (f1 - f0) / eps
            if abs(jac) > 1e-6:
                dx = -np.clip(f0 / jac, -0.2, 0.2)
                if var == "h":
                    h += dx
                elif var == "a":
                    a += dx
                elif var == "r":
                    r += dx
                else:
                    bz += dx
                f = residuals(h, a, r, bz)   # Gauss-Seidel 式立即刷新（实测脚本同款）
    return float(h), float(a), float(r), float(bz)


def _ik_entry(model: mujoco.MjModel, data: mujoco.MjData, k: float) -> tuple[float, float, float, float]:
    """查 IK 表（模块级缓存，所有环境共享同一张表）。"""
    key = round(float(k), 3)
    if key not in _IK_CACHE:
        _IK_CACHE[key] = _ik_solve(model, data, key)
    return _IK_CACHE[key]


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 欧拉角 -> MuJoCo 四元数 (w,x,y,z)。"""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


class T800GetUpEnv(gym.Env):
    """浮动基座双足从深蹲站起并站住的强化学习环境。"""

    metadata = {"render_modes": []}

    def __init__(self, level: int = 3) -> None:
        super().__init__()
        self.model, self.data = tc.load_model()
        self.addr = tc.build_addr(self.model)
        self.addr.kp[self.addr.leg] = KP_LEG   # 深蹲 PD 刚度（其余关节保持原增益）
        self.addr.kd[self.addr.leg] = KD_LEG
        self.leg_idx = self.addr.leg
        self.jnt_range = self.model.jnt_range[self.addr.jid[self.leg_idx]]  # (12,2)

        self.action_space = spaces.Box(-1.0, 1.0, (len(ACTION_LEG_IDX),), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (45,), dtype=np.float32)

        self.frame_skip = tc.FRAME_SKIP
        self.max_steps = MAX_STEPS
        self.level = int(level)
        self._steps = 0
        self._prev_action = np.zeros(len(ACTION_LEG_IDX), dtype=np.float64)
        self._h0 = TARGET_HEIGHT
        self._h_prev = TARGET_HEIGHT
        self._ema_h = TARGET_HEIGHT
        self._best_ema = TARGET_HEIGHT
        self._last_progress = 0
        self._success_count = 0

    def set_level(self, level: int) -> None:
        self.level = int(level)

    # ---- 核心接口 ----

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        k_lo, k_hi = CURRICULUM_K[self.level]
        k = round(float(rng.uniform(k_lo, k_hi)) / 0.05) * 0.05
        k = min(max(k, 0.05), 1.25)
        h, a, r, bz = _ik_entry(self.model, self.data, k)

        q0 = np.zeros(25, dtype=np.float64)
        q0[0] = q0[6] = h
        q0[3] = q0[9] = k
        q0[4] = q0[10] = a
        q0[5] = r
        q0[11] = -r
        q0 += rng.uniform(-0.02, 0.02, size=25)  # 关节小噪声

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = rng.uniform(-0.02, 0.02, size=2)
        self.data.qpos[2] = bz                                  # 足底恰触地（已解好）
        self.data.qpos[3:7] = _euler_to_quat(*rng.uniform(-0.03, 0.03, size=2), rng.uniform(-np.pi, np.pi))
        self.data.qpos[self.addr.qpos] = q0
        mujoco.mj_forward(self.model, self.data)
        self.data.qvel[3:6] += rng.uniform(-0.05, 0.05, size=3)  # 基座角速度扰动

        # 落稳 0.2s：PD 锁持坐姿，避免落地瞬态的 CoP 冲击（研究实测会「发射」脚）
        for _ in range(100):
            tau = tc.pd_torque_fast(self.data, self.addr, q0)
            self.data.ctrl[self.addr.act] = tau
            mujoco.mj_step(self.model, self.data)

        self._steps = 0
        self._prev_action = np.zeros(len(ACTION_LEG_IDX), dtype=np.float64)
        self._q_des_ref = q0.copy()                            # 目标低通状态：从坐姿目标起步
        self._h0 = tc.base_height(self.data)                    # 回合起始髋高（EMA 初始化基准）
        self._h_prev = self._h0
        self._ema_h = self._h0
        self._best_ema = self._h0
        self._last_progress = 0
        self._success_count = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        q_des = np.zeros(25, dtype=np.float64)
        leg_target = np.zeros(12, dtype=np.float64)
        leg_target[ACTION_LEG_IDX] = action * ACTION_HALF_RANGE  # 绝对目标角
        lo = self.jnt_range[:, 0] + 0.03
        hi = self.jnt_range[:, 1] - 0.03
        q_des[self.leg_idx] = np.clip(leg_target, lo, hi)
        # 目标低通（比例式）：所有关节按剩余行程等比例扫向目标、同步到站。
        # 速率上限作用于行程最长的关节。实测：统一速率会让踝提前归零 → CoP 前移翻倒。
        delta = q_des - self._q_des_ref
        max_step = float(np.max(np.abs(delta)))
        if max_step > QDES_MAX_RATE:
            delta = delta * (QDES_MAX_RATE / max_step)
        self._q_des_ref = self._q_des_ref + delta
        q_des = self._q_des_ref

        tau = None
        for _ in range(self.frame_skip):
            tau = tc.pd_torque_fast(self.data, self.addr, q_des)
            self.data.ctrl[self.addr.act] = tau
            mujoco.mj_step(self.model, self.data)

        obs = self._obs()
        reward, terms = self._reward(action, tau)
        terminated, term_info = self._termination_check()

        # 成功：成功带连续 1.5s → 一次性终局奖励 + 提前终止
        if self._success_count >= SUCCESS_STEPS:
            terminated = True
            term_info["success"] = True
            reward += SUCCESS_BONUS

        truncated = (self._steps >= self.max_steps) or (self._steps - self._last_progress >= STUCK_STEPS)
        self._steps += 1
        self._prev_action = action.copy()
        info = {"reward_terms": terms, **term_info}
        return obs, float(reward), terminated, truncated, info

    # ---- 观测 / 奖励 / 终止 ----

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[self.addr.qpos[self.leg_idx]]          # 12 腿关节位置
        qd = self.data.qvel[self.addr.dof[self.leg_idx]]          # 12 腿关节速度
        ang_vel = self.data.qvel[3:6]          # 基座角速度
        lin_vel = self.data.qvel[:3]           # 基座线速度
        gravity = tc.projected_gravity(self.data)
        contact = np.tanh(tc.foot_contacts_fast(self.data, self.addr) / 400.0)
        h = tc.base_height(self.data)
        com_z = float(self.data.subtree_com[0][2])                # 全身质心高
        return np.concatenate([
            q,                                  # 0:12  腿关节位置（=误差，名义为 0）
            0.05 * qd,                          # 12:24 腿速度（缩放）
            self._prev_action,                  # 24:32 上一动作（8 维）
            ang_vel,                            # 32:35 基座角速度
            gravity,                            # 35:38 投影重力
            lin_vel,                            # 38:41 基座线速度
            contact,                            # 41:43 足底接触
            [h - 0.25, com_z - 0.25],           # 43:45 基座高度、质心高度（核心信号）
        ]).astype(np.float32)

    def _reward(self, action: np.ndarray, tau: np.ndarray) -> tuple[float, dict]:
        h = tc.base_height(self.data)
        # progress = 纯 Δh（v4）：马尔可夫、无回合隐藏态。旧版 Δp 归一化（denom=TARGET−h0）
        # 让同一物理状态在不同回合 reward 差 ±5/步（h0 不被观测）→ 近站立态价值失准、
        # 确定性评估大动作自毁（实测 L0 30%<L1 90% 倒挂 + value_loss 尖峰）。纯 Δh 天然按
        # 站起距离等权（每回合总和 = 5×Δh_total），且 Σ5Δh 伸缩和为 0（rise-fall 循环
        # progress 净 0）。
        progress = 5.0 * (h - self._h_prev)

        gravity = tc.projected_gravity(self.data)
        tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))

        # 成功带：EMA 高度 + 倾角 + 双足接触同时满足（EMA 防单帧弹跳伪造）
        self._ema_h += 0.025 * (h - self._ema_h)
        if self._ema_h > self._best_ema + 1e-3:      # stuck 检测：高度 EMA 创新高才刷时间戳
            self._best_ema = self._ema_h
            self._last_progress = self._steps
        fz = tc.foot_contacts_fast(self.data, self.addr)
        in_band = (self._ema_h >= SUCCESS_H and tilt <= SUCCESS_TILT and np.min(fz) > SUCCESS_FZ)

        # v4c-1：带外弱 upright 指引（0.02，h>0.75 门控，独立项）。
        # v4 实测教训：P1 把 upright 全收进带内后，升程失去姿态整形 → 策略学出「全幅狂推」
        # 吸引子（|a|≈2.1-2.8 从第 0 步推到翻倒，L2 全期 0-10%、L3 40-70% 震荡）。深蹲升程
        # 起点 h≈0.84>0.75，弱指引全程活跃，只整形姿态、不影响「站起」本身（收割循环利润
        # ~+8/周期 vs 带内 138，可忽略——评审裁定）。
        upright_guide = (0.02 * float(np.exp(-8.0 * np.sum((gravity - np.array([0, 0, -1])) ** 2)))
                         if h > 0.75 else 0.0)

        # P1：upright/feet 只在成功带内给（v4 配套防 hacking 补丁）。旧版 h>0.75/h>0.7
        # 门控允许「站起收割 → 倒回/蹲回 → 再起」循环在带外白赚 ~0.45/步（progress 伸缩和
        # 为 0、躺地期惩罚≈0），v1 纯 L2 训练实证学坏（mean_ep_return 149→19.5）。带内给 →
        # 收割总量被 150 步成功终局硬性封顶（≤150×0.79+20≈138），与诚实站起持平，循环
        # 无利可图。注意门控含双足>100N：单足离地立刻停止收割，对失败签名（单足卸载）有
        # 直接纠正梯度。
        upright = 0.4 * float(np.exp(-8.0 * np.sum((gravity - np.array([0, 0, -1])) ** 2))) if in_band else 0.0
        contact = np.tanh(fz / 400.0)
        feet = 0.1 * float(np.mean(contact)) if in_band else 0.0

        # v4d-1：带内镇定整形（4 路评审 workflow 裁定权重，2026-11-14）。v4c 实测教训：
        # 0.1/0.05 惩罚只命中收割（0.72-0.74/步）的 1/8.7~1/10.4，是「固定税」而非纠偏
        # 梯度，crank 吸引子纹丝不动（带内 |a|₂ 均值 1.67-2.20、成功=骑线过杆：反事实关掉
        # 成功终局后 11/11 个「本应成功」回合过杆后 4-87 步内全部崩溃）。v4d 按评审二权重：
        # crank 态合计 ≈0.32-0.46/步 = 收割的 47-63%（v4c 的 7.3 倍），crank-vs-quiet 梯度差
        # ≈48-69/回合；带边最坏角落（tilt 0.22、fz→100N、|a|→1）净额 ≈−0.01..+0.02≈0，
        # 不给「避免进带/带边弃守」留口子。单回合上限 ≤206 理论（现实 crank 停留 ≈52-69、
        # quiet ≈5-6），回报动态范围 ≈[−200,+140] = v4c 的 2 倍，远低于 v2 教训的 10 倍线；
        # 全部马尔可夫（本步状态/动作决定）、门控带内、150 步封顶。预登记升级：若成熟期
        # 带内 mean|a| 不降到 <0.5 → 收割三项乘 (1−min(1,mean|a|)) 结构门控。
        action_mag = -0.4 * float(np.mean(np.abs(action))) if in_band else 0.0
        foot_asym = -0.2 * float(abs(fz[0] - fz[1]) / max(fz[0] + fz[1], 1.0)) if in_band else 0.0
        # 只取 8 个动作关节（被动髋roll/yaw 恒≈0 会把罚金稀释 1.5×）；开环 a=0 通路入口
        # q²≈0.03-0.11 → 只付 ≈0.008-0.033/步（带内实收的 ~1%），罚金打在 crank 蹲折姿态上
        q_act = self.data.qpos[self.addr.qpos[ACTION_LEG_IDX]]
        posture_pen = -0.3 * float(np.mean(q_act ** 2)) if in_band else 0.0

        if in_band:
            self._success_count += 1
            success_hold = 0.3
        else:
            success_hold = 0.0
        # v4d 必修修复（防黑评审实测漏洞）：_success_count 必须回合内累计、出带不清零（仅
        # reset 清零）。旧版出带清零使「150 步成功终局」只封顶连续段——策略可 flicker 出带
        # （单足卸载/tilt 擦边/EMA 擦边）重置计数器再入带，带内步数无上限：f=0.9 的 flicker
        # 循环 2000 步 ≈ +1350~1400/回合 = 10× 诚实站起（≈138）。累计计数恢复硬封顶：
        # 任何回合带内步数 ≤150 即触发成功终局，收割 ≤150×0.79+20≈138，循环无利可图。
        # 语义：成功从「连续 150 步」放松为「累计 150 步」——实测真站起策略均单次入带，
        # 无行为差异；对 flicker 策略则彻底封死。

        qd = self.data.qvel[self.addr.dof[self.leg_idx]]
        ang_vel = self.data.qvel[3:6]
        lin_vel = self.data.qvel[:3]
        terms = {
            "progress": progress,
            "upright_guide": upright_guide,
            "upright": upright,
            "feet": feet,
            "success_hold": success_hold,
            "action_mag": action_mag,
            "foot_asym": foot_asym,
            "posture_pen": posture_pen,
            # 注意：不要加重的低姿态惩罚——实测 -0.3/步 会把回报动态范围扩大 10 倍
            # （单回合 -1000+），价值函数失稳、策略在「学会-遗忘」间震荡（v2 教训）
            "torque": -2e-6 * float(np.mean(tau[self.leg_idx] ** 2)),
            "action_rate": -0.01 * float(np.mean((action - self._prev_action) ** 2)),
            "joint_velocity": -0.002 * float(np.mean(qd ** 2)),
            "lateral_ang_vel": -0.15 * float(ang_vel[0] ** 2 + ang_vel[2] ** 2),
            "base_lin_vel": -0.1 * float(np.sum(lin_vel ** 2)),
        }
        self._h_prev = h
        return float(sum(terms.values())), terms

    def _termination_check(self) -> tuple[bool, dict]:
        d = self.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            return True, {"nan": True}
        if abs(d.qpos[0]) > 1.5 or abs(d.qpos[1]) > 1.5 or d.qpos[2] > 1.6:
            return True, {"out_of_bounds": True}      # 垃圾姿态可飞 11m，兜底
        q = d.qpos[self.addr.qpos[self.leg_idx]]
        lo = self.jnt_range[:, 0] - 0.03
        hi = self.jnt_range[:, 1] + 0.03
        if np.any(q < lo) or np.any(q > hi):
            return True, {"joint_limit": True}
        return False, {}


if __name__ == "__main__":
    # 自检 1：随机动作 rollout，验证形状、无 NaN、终止与 reset 正常
    env = T800GetUpEnv(level=3)
    obs, _ = env.reset()
    print(f"obs shape = {obs.shape}, action shape = {env.action_space.shape}")
    print(f"reset: base_height={obs[43] + 0.25:.3f}m  com_z={obs[44] + 0.25:.3f}m")
    for ep in range(3):
        obs, _ = env.reset()
        ep_reward = 0.0
        for t in range(env.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            assert obs.shape == (45,), obs.shape
            assert np.isfinite(obs).all(), "obs 出现 NaN"
            if terminated or truncated:
                break
        print(f"episode {ep}: length={t + 1}, reward={ep_reward:.1f}, "
              f"terminated={terminated}, truncated={truncated}, info={ {k: v for k, v in info.items() if k != 'reward_terms'} }")

    # 自检 2：action=0（目标=站立）从深蹲起步 300 步——必须站起并站住（环境物理通路验收）
    env.set_level(2)   # 深蹲档
    obs, _ = env.reset()
    h0 = obs[43] + 0.25
    for _ in range(300):
        obs, _, terminated, truncated, info = env.step(np.zeros(len(ACTION_LEG_IDX)))
        if terminated or truncated:
            break
    h_end = obs[43] + 0.25
    g = tc.projected_gravity(env.data)
    tilt_end = float(np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0))))
    print(f"\naction=0 开环检查：h0={h0:.3f}m -> 300 步后 base_height={h_end:.3f}m, "
          f"tilt={tilt_end:.1f}° (terminated={terminated}, truncated={truncated}, info={ {k: v for k, v in info.items() if k != 'reward_terms'} })")
    assert h_end > 0.90 and tilt_end < 10.0, "开环站起失败：环境物理通路异常"

    # 自检 3：吞吐量实测
    env.set_level(3)
    env.reset()
    n_bench = 500
    t0 = time.time()
    for _ in range(n_bench):
        env.step(np.zeros(len(ACTION_LEG_IDX)))
        if env._steps >= env.max_steps:
            env.reset()
    dt = time.time() - t0
    print(f"\n吞吐量: {n_bench / dt:.0f} env.step/s（每步含 {env.frame_skip} 个物理子步）")
    print("环境自检通过。")
