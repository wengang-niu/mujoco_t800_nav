"""10 — T800「地面折叠蹲→深蹲交接」Gymnasium 环境（地面站起可行性定论实验）。

背景（2026-11-14 六路探索 workflow 实测，全部经复跑确认）：
- 深蹲→站立（07/08/09）已验收：v4d ckpt1800 L2 96.5%/L3 99.0%，从 k≤1.2 深蹲族起步。
- 地面站起开环可行性探索结论：六种地面姿态（腿前伸坐/折叠坐/仰卧/俯卧/侧卧/跪）开环
  轨迹全部失败（最佳 max_z：side 0.692 / folded 0.627 / pike 桥 0.624，交接需 0.87）。
  诊断高度收敛：① 踝 35 N·m 帽（CoP 权限 ±8.4cm）是决定性瓶颈；② 无腰俯仰关节
  （J12 仅 yaw）——髋屈反作用把骨盆后翻；③ 质量层级（躯干 46.5kg vs 腿 19.2kg）腿力矩
  优先转腿不转躯干；④ 踝背屈行程 ±39° 不够折叠姿态足平贴地；⑤ 足锚摩擦 136N < 腿链
  拉力 300-550N。
- 但存在两个真实升程：手锚 pike 桥稳定升 0.10→0.62；折叠蹲膝先行受控升 0.53→0.63
  （之后足跟滚动后仰崩塌——动量时机问题）。真地面折叠坐（骨盆 0.128m）对本机器人
  几何不可达（膝最大 135°，骨盆物理下限 ≈0.41-0.49m）；最深可达地面姿态 = 折叠蹲
  tuck（髋 -1.40、膝 2.30、踝 -0.65、骨盆 ≈0.55，双足踩前足盒角点）。

本环境设计（定论实验）：
- reset 从折叠蹲 tuck 起步（最深可达地面姿态，确定性）+ 小噪声，落稳 0.2s。
- 动作 12 维：07 的 8 腿关节（绝对目标角）+ 4 臂关节（肩 pitch ×2、肘 pitch ×2 绝对
  目标角）——臂是可行性研究证实的动量/锚定执行器（手锚 pike 是唯一无滚动升程）。
- 成功 = 深蹲交接：EMA 骨盆高 ≥0.87 且 tilt ≤0.35 且双足力 >200N（地面姿态足力阈值
  从 07 的 100N 提高到 200N——折叠小腿压足自接触会伪造 125/94N 读数）累计 150 步
  → +20 提前终局（计数器出带不清零，07 同款防 flicker）。
- 奖励 v3（run 3 裁定，2026-11-14 终审 workflow wf_8a7bf9bc-06a）：「记录势」
  r1 权重 10→12；r2 手锚信用整项删除（探针实证 tuck 手高 0.63 够不到地——结构性
  死项）；新增 r9 预带区块（买「持续预带段」而非瞬时记录：pre_active = ema_z≥0.72 ∧
  min fz>150 ∧ com_fwd≥0 ∧ pre_cnt<150；r9 = pre_active×[0.10 + 0.08·exp(−8‖g−ẑ‖²)
  + 0.12·min(1,exp(−0.5((fwd−0.02)/0.03)²)) − 0.10·mean|a|]；步上限 0.30、回合
  封顶 150 步 ≤45；pre_cnt 回合内累计不清零）。r3/r4 引导、带内 v4d 块、地板税
  原样。目标角低通改 z 调度速率：z<0.80 用 0.012、z≥0.80 用 0.006（永不用 0.018）。
  obs 仍 62 维。reset 新增 options["pitch"]（squat 侧查 pitch=0 = v4d 平衡流形）。
- 终止：NaN/出界/关节超限/stuck（500 步 m_ema 无创新）/成功/2000 步超时。不因
  倒地终止——允许摔倒重试（07 同款）。
"""
from __future__ import annotations

import time

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import t800_common as tc

# 动作关节：腿 8（07 同款）+ 臂 4（肩 pitch ×2、肘 pitch ×2）
ACTION_LEG_IDX = np.array([0, 3, 4, 5, 6, 9, 10, 11], dtype=int)
ACTION_ARM_IDX = np.array([13, 16, 18, 21], dtype=int)   # 肩P_L 肘P_L 肩P_R 肘P_R
ACTION_HALF_RANGE = np.concatenate([
    np.array([2.79, 1.31, 0.68, 0.26, 2.79, 1.31, 0.68, 0.26]),   # 腿（07 同款）
    np.array([1.57, 1.14, 1.57, 1.14]),                          # 臂（肩 pitch ±90°、肘 ±65°）
])

KP_LEG, KD_LEG = 800.0, 40.0   # 腿（07 实测）
KP_ARM, KD_ARM = 400.0, 20.0   # 臂（探索探针用值，支撑刚度足够）
MAX_STEPS = 2000
SUCCESS_STEPS = 150            # 交接带累计 150 步（1.5s，出带不清零）→ 提前终局
SUCCESS_BONUS = 20.0
STUCK_STEPS = 500
QDES_MAX_RATE = 0.006           # z≥0.80 的目标低通速率（07 实测值）
QDES_RATE_LOW = 0.012           # z<0.80 的升程速率（run 3 裁定 z 调度，低姿态扫快一倍）
QDES_RATE_SWITCH_Z = 0.80       # 速率切换高度（m）
SUCCESS_Z = 0.87               # 深蹲交接高度（k=1.2 pelvis 高度）
SUCCESS_TILT = 0.35
SUCCESS_FZ = 200.0             # 地面姿态足力阈值（防折叠小腿压足自接触伪造）
TARGET_HEIGHT = tc.TARGET_HEIGHT

# 折叠蹲 tuck 起始姿态（探索 workflow folded agent v2 配置，实测确定性落稳 z≈0.53-0.55）
TUCK_Q = {
    "J00_HIP_PITCH_L": -1.40, "J03_KNEE_PITCH_L": 2.30,
    "J04_ANKLE_PITCH_L": -0.65, "J05_ANKLE_ROLL_L": 0.17,
    "J06_HIP_PITCH_R": -1.40, "J09_KNEE_PITCH_R": 2.30,
    "J10_ANKLE_PITCH_R": -0.65, "J11_ANKLE_ROLL_R": -0.17,
    "J13_SHOULDER_PITCH_L": -0.6, "J16_ELBOW_PITCH_L": -0.8,
    "J18_SHOULDER_PITCH_R": -0.6, "J21_ELBOW_PITCH_R": -0.8,
}
TUCK_BASE_PITCH = 0.15          # 躯干前倾（agent 实测的稳定偏置）
TUCK_Z0 = 0.550

# ---- 奖励 v2 超参（记录势重构，总合成定稿）----
EMA_ALPHA = 0.025              # EMA 滤波（100Hz 策略步，τ≈0.4s，防弹射甩高收割）
RECORD_CAP = SUCCESS_Z         # 记录 m_ema 封顶 0.87（带内不再付 r1）
R1_W = 12.0                    # r1 记录信用权重（run 3 裁定：10→12）
R1_GATE_MIN_FZ = 0.0           # r1 支付门槛（0=关；判据③触发时置 50.0 防弹射收割）
GUIDE_Z = 0.65                 # r3/r4 引导门槛（原 0.60 上提）
GUIDE_FZ = 50.0                # r3/r4 足载门槛（防足离地高姿态收割，评审必改项）
R3_W = 0.03                    # r3 CoM 前移守线（高斯带：踝前 4±3cm）
COM_FWD_TARGET = 0.04          # CoM 相对双足踝中点前移目标（m；足半长 5.7cm → 前中部）
COM_FWD_SIGMA = 0.03           # tuck 出生 CoM 已在踝前 ~10cm：语义是「过墙相别后滚」，
                               # 而非前移引导（评审实证 com_fwd 出生即饱和 1.64）
R4_W = 0.02                    # r4 upright 弱引导（仅记录创新步）
FLOOR_Z = 0.40                 # 地板税门槛（低于坐姿物理下限 0.41-0.49 = 纯躺区）
FLOOR_TAX = -0.02              # 地板税（堵躺地；真实崩塌穿越段收 −0.2~−1.0，F1 监测）

# ---- r9 预带区块（run 3 裁定新增，替代已删除的 r2 手锚死项）----
PRE_Z = 0.72                   # pre_active 高度门槛（EMA 骨盆高）
PRE_FZ = 150.0                 # pre_active 双足力门槛
PRE_CNT_MAX = 150              # r9 付款步数封顶（之后静默；步上限 0.30 → 回合 ≤45）
PRE_FWD_TARGET = 0.02          # CoM 前移高斯中心（m，相对双足踝中点）
PRE_FWD_SIGMA = 0.03           # CoM 前移高斯带宽（m）
R9_BASE = 0.10                 # 基础付款
R9_UP_W = 0.08                 # upright 贴合权重
R9_FWD_W = 0.12                # CoM 前移贴合权重
R9_ACT_PEN = 0.10              # 动作幅惩罚（抑制预带内 crank 姿态）

# 课程：squat 起步（Stage A 带内技能 / iter800 救援）。k=1.2 是 07 v4d 验收过的
# 课程参数（07 docstring 记髋 0.84；本环境实测落稳 z0≈0.857——带边 0.87 之下
# 1.3cm，教「进带+保持」；实测 k=1.0 落稳 0.889 已在带内、不教进带，故不用）。
SQUAT_COURSE_K = 1.2


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


class T800GetUpGroundEnv(gym.Env):
    """从折叠蹲 tuck 起步，站到深蹲交接带的地面站起环境（定论实验）。"""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.model, self.data = tc.load_model()
        self.addr = tc.build_addr(self.model)
        self.addr.kp[self.addr.leg] = KP_LEG
        self.addr.kd[self.addr.leg] = KD_LEG
        self.addr.kp[ACTION_ARM_IDX] = KP_ARM
        self.addr.kd[ACTION_ARM_IDX] = KD_ARM
        self.leg_idx = self.addr.leg
        self.jnt_range = self.model.jnt_range[self.addr.jid]
        self.hand_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "LINK_WRIST_END_L")
        self.hand_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "LINK_WRIST_END_R")
        assert self.hand_l_id >= 0 and self.hand_r_id >= 0, "手部 body 名无效"
        self.foot_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "LINK_FOOT_L")
        self.foot_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "LINK_FOOT_R")
        assert self.foot_l_id >= 0 and self.foot_r_id >= 0, "足部 body 名无效"

        self.act_idx = np.concatenate([ACTION_LEG_IDX, ACTION_ARM_IDX])
        self.action_space = spaces.Box(-1.0, 1.0, (len(self.act_idx),), dtype=np.float32)
        # 59 → 62：追加 (ema_z, m_ema 记录, com_fwd)。奖励是 (obs_t, obs_{t+1}) 的
        # 确定性函数——严格马尔可夫（07 v3 L0 倒挂机制的对治；obs_t 含更新前记录，
        # 交叉/创新事件由连续两观测完全重建）。
        self.observation_space = spaces.Box(-10.0, 10.0, (62,), dtype=np.float32)

        self.frame_skip = tc.FRAME_SKIP
        self.max_steps = MAX_STEPS
        self._steps = 0
        self._prev_action = np.zeros(len(self.act_idx), dtype=np.float64)
        self._z0 = TUCK_Z0
        self._z_prev = TUCK_Z0
        self._ema_z = TUCK_Z0
        self._best_ema = TUCK_Z0
        self._last_progress = 0
        self._success_count = 0
        self._pre_cnt = 0               # r9 付款计数（回合内累计不清零，07 同款防 flicker）

    # ---- 核心接口 ----

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        mode = (options or {}).get("start", "tuck")
        # run 3：options["pitch"] 显式覆盖基座俯仰。squat 侧查 pitch=0（07 同款 v4d
        # 平衡流形）；None = 默认 TUCK_BASE_PITCH（tuck 课 +0.15 稳定偏置，实测必保）。
        pitch_opt = (options or {}).get("pitch", None)
        base_pitch = TUCK_BASE_PITCH if pitch_opt is None else float(pitch_opt)

        if mode == "tuck":
            q0 = np.zeros(25, dtype=np.float64)
            for n, v in TUCK_Q.items():
                q0[tc.JOINT_NAMES.index(n)] = v
            q0 += rng.uniform(-0.02, 0.02, size=25)
            bz = TUCK_Z0
        elif mode == "squat":
            # 课程起点：07 的足平贴地对称深蹲 IK（k=1.0 → 髋≈0.86，带边之下）。
            # 评审实测教训：07 的 _ik_entry 有模块级缓存且就地改写传入 data——
            # 此处改用 _ik_solve + 独立 scratch MjData，绝不碰 self.data。
            import importlib
            ik = importlib.import_module("07_get_up")
            scratch = mujoco.MjData(self.model)
            h, a, r, bz = ik._ik_solve(self.model, scratch, SQUAT_COURSE_K)
            q0 = np.zeros(25, dtype=np.float64)
            for n, v in (("J00_HIP_PITCH_L", h), ("J06_HIP_PITCH_R", h),
                         ("J03_KNEE_PITCH_L", SQUAT_COURSE_K), ("J09_KNEE_PITCH_R", SQUAT_COURSE_K),
                         ("J04_ANKLE_PITCH_L", a), ("J10_ANKLE_PITCH_R", a),
                         ("J05_ANKLE_ROLL_L", r), ("J11_ANKLE_ROLL_R", -r)):
                q0[tc.JOINT_NAMES.index(n)] = v
            q0 += rng.uniform(-0.01, 0.01, size=25)
        elif mode == "kneel":
            raise NotImplementedError("跪姿预设（z≈0.69）未入仓：六姿态探索为临时脚本，"
                                      "需先移植数值再开课（评审确认 scene/ 只有房间渲染）")
        else:
            raise ValueError(f"未知起始模式: {mode}")

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = rng.uniform(-0.02, 0.02, size=2)
        self.data.qpos[2] = bz
        # 基座俯仰：tuck 用 +0.15 稳定偏置（pitch-0 tuck 是死起点——z0 0.498 不升，
        # 探针实证）；squat 默认同 +0.15 教「进带」，侧查用 options["pitch"]=0。
        self.data.qpos[3:7] = _euler_to_quat(rng.uniform(-0.02, 0.02),
                                             base_pitch + rng.uniform(-0.02, 0.02),
                                             rng.uniform(-0.02, 0.02))
        self.data.qpos[7:] = q0
        mujoco.mj_forward(self.model, self.data)
        self.data.qvel[3:6] += rng.uniform(-0.03, 0.03, size=3)

        # 落稳 0.2s：PD 锁持起始姿态（tuck/squat 同款探针流程）
        for _ in range(100):
            self.data.ctrl = tc.pd_torque_fast(self.data, self.addr, q0)
            mujoco.mj_step(self.model, self.data)

        self._steps = 0
        self._prev_action = np.zeros(len(self.act_idx), dtype=np.float64)
        self._q_des_ref = q0.copy()
        self._z0 = tc.base_height(self.data)
        self._z_prev = self._z0
        self._ema_z = self._z0
        self._best_ema = min(self._z0, RECORD_CAP)   # 记录封顶：squat 课 z0>0.87 时不回落
        self._last_progress = 0
        self._success_count = 0
        self._pre_cnt = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        q_des = np.zeros(25, dtype=np.float64)
        q_des[self.act_idx] = action * ACTION_HALF_RANGE
        lo = self.jnt_range[:, 0] + 0.03
        hi = self.jnt_range[:, 1] - 0.03
        q_des = np.clip(q_des, lo, hi)
        # 比例式目标低通（07 同款：按剩余行程等比例扫向目标、关节同步到站）。
        # run 3 裁定 z 调度：低姿态（升程段）0.012 扫快一倍，z≥0.80 恢复 0.006。
        rate = (QDES_RATE_LOW if tc.base_height(self.data) < QDES_RATE_SWITCH_Z
                else QDES_MAX_RATE)
        delta = q_des - self._q_des_ref
        max_step = float(np.max(np.abs(delta)))
        if max_step > rate:
            delta = delta * (rate / max_step)
        self._q_des_ref = self._q_des_ref + delta
        q_des = self._q_des_ref

        tau = None
        for _ in range(self.frame_skip):
            tau = tc.pd_torque_fast(self.data, self.addr, q_des)
            self.data.ctrl = tau
            mujoco.mj_step(self.model, self.data)

        obs = self._obs()
        reward, terms = self._reward(action, tau)
        terminated, term_info = self._termination_check()

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
        q_leg = self.data.qpos[self.addr.qpos[self.leg_idx]]      # 12
        qd_leg = self.data.qvel[self.addr.dof[self.leg_idx]]      # 12
        q_arm = self.data.qpos[self.addr.qpos[ACTION_ARM_IDX]]    # 4
        qd_arm = self.data.qvel[self.addr.dof[ACTION_ARM_IDX]]    # 4
        ang_vel = self.data.qvel[3:6]
        lin_vel = self.data.qvel[:3]
        gravity = tc.projected_gravity(self.data)
        contact = np.tanh(tc.foot_contacts_fast(self.data, self.addr) / 400.0)
        h = tc.base_height(self.data)
        com_z = float(self.data.subtree_com[0][2])
        # 手部高度（接触代理：手 z < 0.15 ≈ 撑地）
        hand_l = float(self.data.xpos[self.hand_l_id][2])
        hand_r = float(self.data.xpos[self.hand_r_id][2])
        # CoM 相对双足踝中点的前移量（归一 0.06m；r3 的状态）
        com_x = float(self.data.subtree_com[0][0])
        foot_x = float(np.mean([self.data.xpos[self.foot_l_id][0],
                                self.data.xpos[self.foot_r_id][0]]))
        com_fwd = float(np.clip((com_x - foot_x) / 0.06, -3.0, 3.0))
        # 记录势观测（obs 在奖励更新前快照 ⇒ 携带上一策略步的 ema/记录值；
        # 本步奖励由 (obs_t, obs_{t+1}) 完全重构，严格马尔可夫）
        return np.concatenate([
            q_leg,                              # 0:12
            0.05 * qd_leg,                      # 12:24
            self._prev_action,                  # 24:36 上一动作（12 维）
            ang_vel,                            # 36:39
            gravity,                            # 39:42
            lin_vel,                            # 42:45
            contact,                            # 45:47 足底接触
            [h - 0.25, com_z - 0.25],           # 47:49 骨盆高、质心高
            q_arm,                              # 49:53 臂位置
            0.05 * qd_arm,                      # 53:57 臂速度
            [hand_l, hand_r],                   # 57:59 手部高度
            [self._ema_z - 0.25,                # 59 EMA 骨盆高（滞后滤波）
             self._best_ema - 0.25,             # 60 回合内 EMA 记录 m_ema（单调、封顶 0.87）
             com_fwd],                          # 61 CoM 前移（归一）
        ]).astype(np.float32)

    def _reward(self, action: np.ndarray, tau: np.ndarray) -> tuple[float, dict]:
        """奖励 v3（run 3 裁定）：
        r = r1(记录信用 12，可加双足力门) + r9(预带区块) + r3/r4(创新步引导)
            + r5/r6(带内 v4d 原样) + r7 地板税 + r8 微罚
        核心不变式：Σr1 = 12×(m_T−m_0) ≥ 0 恒成立（m_ema 单调、封顶 0.87）；
        升程只赚、回落免税、回升不重付、躺地 0 收入、弹射被 EMA τ=0.4s 低通。
        r9 买「持续预带段」（ema_z≥0.72 ∧ 双足载>150 ∧ CoM 过踝 ∧ 贴 upright ∧
        小动作），回合封顶 150 步——把「瞬时过带」换成「预带栖息」的学费。
        更新顺序：先 EMA → 再记录 → 再计分（obs 已在上一步快照更新前值）。
        """
        z = tc.base_height(self.data)
        gravity = tc.projected_gravity(self.data)
        tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
        fz = tc.foot_contacts_fast(self.data, self.addr)

        # ---- 0) EMA → 记录 → 创新判定（stuck 计时沿用记录创新语义）----
        self._ema_z += EMA_ALPHA * (z - self._ema_z)
        m_old = self._best_ema
        self._best_ema = min(max(m_old, self._ema_z), RECORD_CAP)
        if self._best_ema > m_old + 1e-3:
            self._last_progress = self._steps
        delta_rec = max(0.0, self._best_ema - m_old)     # Δ⁺m_ema，构造上 ≥0
        innov = delta_rec > 0.0                          # 本步 = 记录创新步

        # ---- 1) r1 记录信用（替代 5Δz；全时崩塌豁免，物理墙封顶）。
        #         run 3：权重 10→12；R1_GATE_MIN_FZ>0 时加双足力门槛（判据③防弹射）----
        r1 = R1_W * delta_rec
        if R1_GATE_MIN_FZ > 0.0 and np.min(fz) <= R1_GATE_MIN_FZ:
            r1 = 0.0

        # ---- 2) r9 预带区块（run 3 裁定新增，替代已删除的 r2 手锚死项）：
        #         买「持续预带段」而非瞬时记录——ema_z≥0.72 ∧ 双足载>150 ∧
        #         CoM 已过踝中点（fwd≥0）∧ pre_cnt<150；付 upright 贴合 + CoM
        #         前移高斯带 + 基础款，扣动作幅。步上限 0.30（=0.10+0.08+0.12）、
        #         回合封顶 150 步 ≤45。pre_cnt 回合内累计不清零（防 flicker 重置
        #         收割，07 同款）----
        com_x = float(self.data.subtree_com[0][0])
        foot_x = float(np.mean([self.data.xpos[self.foot_l_id][0],
                                self.data.xpos[self.foot_r_id][0]]))
        fwd = com_x - foot_x
        pre_active = (self._ema_z >= PRE_Z and np.min(fz) > PRE_FZ
                      and fwd >= 0.0 and self._pre_cnt < PRE_CNT_MAX)
        if pre_active:
            self._pre_cnt += 1
            r9 = (R9_BASE
                  + R9_UP_W * float(np.exp(-8.0 * np.sum((gravity - np.array([0, 0, -1])) ** 2)))
                  + R9_FWD_W * min(1.0, float(np.exp(-0.5 * ((fwd - PRE_FWD_TARGET) / PRE_FWD_SIGMA) ** 2)))
                  - R9_ACT_PEN * float(np.mean(np.abs(action))))
        else:
            r9 = 0.0

        # ---- 3) r3 CoM 前移守线（高斯带 4±3cm；tuck 出生 CoM 已在踝前 ~10cm，
        #         语义为「过墙相别后滚」，仅创新步付 → 墙边静持农场归零）----
        r3 = (R3_W * float(np.exp(-0.5 * ((fwd - COM_FWD_TARGET) / COM_FWD_SIGMA) ** 2))
              if (innov and z > GUIDE_Z and np.min(fz) > GUIDE_FZ) else 0.0)

        # ---- 4) r4 upright 弱引导（仅创新步付，防无进度悬停收割）----
        r4 = (R4_W * float(np.exp(-8.0 * np.sum((gravity - np.array([0, 0, -1])) ** 2)))
              if (innov and z > GUIDE_Z and np.min(fz) > GUIDE_FZ) else 0.0)

        # ---- 5/6) 带内收割 + 镇定整形（v4d 原样，评审裁定一字不改）----
        in_band = (self._ema_z >= SUCCESS_Z and tilt <= SUCCESS_TILT and np.min(fz) > SUCCESS_FZ)
        upright = 0.4 * float(np.exp(-8.0 * np.sum((gravity - np.array([0, 0, -1])) ** 2))) if in_band else 0.0
        contact = np.tanh(fz / 400.0)
        feet = 0.1 * float(np.mean(contact)) if in_band else 0.0
        action_mag = -0.4 * float(np.mean(np.abs(action))) if in_band else 0.0
        foot_asym = -0.2 * float(abs(fz[0] - fz[1]) / max(fz[0] + fz[1], 1.0)) if in_band else 0.0
        q_act = self.data.qpos[self.addr.qpos[self.act_idx]]
        posture_pen = -0.3 * float(np.mean(q_act ** 2)) if in_band else 0.0

        if in_band:
            self._success_count += 1
            success_hold = 0.3
        else:
            success_hold = 0.0
        # 成功计数：回合内累计、出带不清零（07 同款防 flicker，reset 才清零）

        # ---- 7) 地板税（堵躺地；真实崩塌穿越段收 −0.2~−1.0，F1 监测其抑制量）----
        r7 = FLOOR_TAX if z < FLOOR_Z else 0.0

        # ---- 8) 微惩罚（不变）----
        qd = self.data.qvel[self.addr.dof[self.leg_idx]]
        ang_vel = self.data.qvel[3:6]
        lin_vel = self.data.qvel[:3]
        terms = {
            "r1_record": r1,
            "r9_pre_band": r9,
            "r3_com_fwd": r3,
            "r4_upright_guide": r4,
            "upright": upright,
            "feet": feet,
            "success_hold": success_hold,
            "action_mag": action_mag,
            "foot_asym": foot_asym,
            "posture_pen": posture_pen,
            "floor_tax": r7,
            "torque": -2e-6 * float(np.mean(tau[self.act_idx] ** 2)),
            "action_rate": -0.01 * float(np.mean((action - self._prev_action) ** 2)),
            "joint_velocity": -0.002 * float(np.mean(qd ** 2)),
            "lateral_ang_vel": -0.15 * float(ang_vel[0] ** 2 + ang_vel[2] ** 2),
            "base_lin_vel": -0.1 * float(np.sum(lin_vel ** 2)),
        }
        self._z_prev = z
        return float(sum(terms.values())), terms

    def _termination_check(self) -> tuple[bool, dict]:
        d = self.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            return True, {"nan": True}
        if abs(d.qpos[0]) > 1.5 or abs(d.qpos[1]) > 1.5 or d.qpos[2] > 1.6:
            return True, {"out_of_bounds": True}
        q = d.qpos[self.addr.qpos[self.leg_idx]]
        # 松弛带 0.05（07 用 0.03）：tuck 起始姿态本身贴机械限位休息（落稳实测
        # 踝 pitch −0.70/限 −0.68、膝 2.36/限 2.35、踝 roll 0.178/限 0.17），
        # 0.03 会把 ~20% 回合变成重置彩票（落稳即贴带边、接触一推就早夭）。
        # 0.05 仍能在关节真正穿限 0.09 rad 时终止（机械绑定态）。
        lo = self.jnt_range[self.leg_idx, 0] - 0.05
        hi = self.jnt_range[self.leg_idx, 1] + 0.05
        if np.any(q < lo) or np.any(q > hi):
            return True, {"joint_limit": True}
        return False, {}


if __name__ == "__main__":
    # 自检 1：随机动作 rollout
    env = T800GetUpGroundEnv()
    obs, _ = env.reset()
    print(f"obs shape = {obs.shape}, action shape = {env.action_space.shape}")
    print(f"reset: base_height={obs[47] + 0.25:.3f}m  com_z={obs[48] + 0.25:.3f}m")
    for ep in range(3):
        obs, _ = env.reset()
        ep_reward = 0.0
        for t in range(env.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            assert obs.shape == (62,), obs.shape
            assert np.isfinite(obs).all(), "obs 出现 NaN"
            if terminated or truncated:
                break
        print(f"episode {ep}: length={t + 1}, reward={ep_reward:.1f}, "
              f"terminated={terminated}, truncated={truncated}, "
              f"info={ {k: v for k, v in info.items() if k != 'reward_terms'} }")

    # 自检 2：持 tuck 动作 150 步（动作→目标角→PD 往返验证；膝目标被半行程
    # 截断到 1.31，低通从 tuck 的 2.30 扫下约需 165 步，150 步内应仍近似持姿）
    obs, _ = env.reset()
    q0 = np.zeros(25)
    for n, v in TUCK_Q.items():
        q0[tc.JOINT_NAMES.index(n)] = v
    a_hold = np.clip(q0[env.act_idx] / ACTION_HALF_RANGE, -1.0, 1.0)
    z_start = obs[47] + 0.25
    for _ in range(150):
        obs, _, terminated, truncated, _ = env.step(a_hold)
        if terminated or truncated:
            break
    print(f"\n锁持检查：z {z_start:.3f} → {obs[47] + 0.25:.3f}m "
          f"(terminated={terminated}, truncated={truncated}, 无 NaN ✓)")

    # 自检 2b：squat 课程起点可落稳（记录信用应从 ≈0 起步，无隐藏分母/无 NaN；
    # k=1.2 应实测 z0≈0.84 带边之下——贴带课程起点，07 v4d 验收参数）
    obs, _ = env.reset(options={"start": "squat"})
    print(f"squat 课程起点：z0={obs[47] + 0.25:.3f}m ema={obs[59] + 0.25:.3f}m "
          f"record={obs[60] + 0.25:.3f}m（应≈髋 0.84、带边 0.87 之下）")

    # 自检 3：吞吐量
    env.reset()
    n_bench = 500
    t0 = time.time()
    for _ in range(n_bench):
        env.step(np.zeros(12))
        if env._steps >= env.max_steps:
            env.reset()
    dt = time.time() - t0
    print(f"\n吞吐量: {n_bench / dt:.0f} env.step/s（每步含 {env.frame_skip} 个物理子步）")
    print("环境自检通过。")
