"""仿真工具 + Walker 编排：观测 → 策略推理 → 目标角 → PD → mj_step。

数据流（每步一个控制周期）：
  obs(72,) → 15 帧历史 → full(1083,) → action(22,) → q_des(25,) → PD×FRAME_SKIP
"""
from __future__ import annotations

import pathlib

import mujoco
import numpy as np

from config import Config
from policy import MnnPolicy


# ---------- 仿真工具 ----------

def load_model(model_path: pathlib.Path):
    model = mujoco.MjModel.from_xml_path(str(model_path))  # 解析 XML 得到只读模型结构
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)                       # 重置到 XML 默认初始状态
    mujoco.mj_forward(model, data)                         # 前向算一次，初始化派生量
    return model, data


def build_addresses(model, joint_names: list):
    qadr = np.array([model.joint(n).qposadr[0] for n in joint_names], dtype=int)
    dadr = np.array([model.joint(n).dofadr[0] for n in joint_names], dtype=int)
    aid = np.array([model.actuator("motor_" + n).id for n in joint_names], dtype=int)
    tau_lim = model.actuator_ctrlrange[aid][:, 1].copy()   # 执行器力矩上限
    return {"qadr": qadr, "dadr": dadr, "aid": aid, "tau_lim": tau_lim}


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pd_step(model, data, addr, q_des, kp, kd):
    """一次 PD 力矩计算 + 一个物理步；力矩按执行器上限截断。"""
    q = data.qpos[addr["qadr"]]
    qd = data.qvel[addr["dadr"]]
    tau = kp * (q_des - q) + kd * (0.0 - qd)
    tau = np.clip(tau, -addr["tau_lim"], addr["tau_lim"])
    data.ctrl[addr["aid"]] = tau
    mujoco.mj_step(model, data)


# ---------- Walker ----------

class Walker:
    """把「观测 1083 → MNN 推理 → 动作 22 → 目标角 → PD → mj_step」串成一步。"""

    def __init__(self, cfg: Config, model=None, data=None):
        self.cfg = cfg
        self.model, self.data = ((model, data) if model is not None
                                 else load_model(cfg.model_path))
        self.addr = build_addresses(self.model, cfg.joint_names)
        self.policy = MnnPolicy(cfg.policy_path)
        self.active_idx = cfg.active_idx
        self.reset()

    def reset(self):
        cfg = self.cfg
        mujoco.mj_resetData(self.model, self.data)        # 状态清零回默认
        self.data.qpos[2] = cfg.base_z                    # 浮基抬到初始高度
        self.data.qpos[self.addr["qadr"]] = cfg.default_q # 25 关节摆到默认站姿
        self.data.qvel[:] = 0.0                           # 速度清零（站稳起步）
        mujoco.mj_forward(self.model, self.data)          # 前向算一遍派生量
        self.prev_action = np.zeros(len(self.active_idx)) # 上一动作清零(22,)
        self.history = [self._obs() for _ in range(cfg.history)]  # 预填历史

    def _obs(self):
        cfg = self.cfg
        q = self.data.qpos[self.addr["qadr"]]                     # (25,) rad
        qd = self.data.qvel[self.addr["dadr"]]                   # (25,) rad/s
        R = quat_to_rot(self.data.qpos[3:7])                     # 姿态四元数 → 旋转矩阵
        # 世界系重力方向投影到机体坐标系：反映机身俯仰/横滚姿态
        proj_g = R.T @ np.array([0.0, 0.0, -1.0])                # (3,)
        ang = self.data.sensor("imu_angular_velocity").data.copy()  # (3,) rad/s
        obs = np.concatenate([
            q[self.active_idx] - cfg.default_q[self.active_idx], # 22 关节角偏差
            qd[self.active_idx],                                 # 22 关节速度
            self.prev_action,                                    # 22 上一动作
            ang,                                                 # 3  机体角速度
            proj_g,                                              # 3  重力投影
        ])                                                       # 共 72 维
        return np.clip(obs * cfg.obs_scale, -cfg.obs_clip, cfg.obs_clip)

    def step(self, command=(0.2, 0.0, 0.0), kp_ratio=1.0):
        cfg = self.cfg
        obs = self._obs()                                          # 当前帧观测
        self.history.append(obs)                                   # 维护历史窗口
        self.history.pop(0)
        full = np.concatenate(self.history + [np.asarray(command) * cfg.cmd_scale])  # 1083
        action = np.clip(self.policy.infer(full), -cfg.action_clip, cfg.action_clip)
        q_des = cfg.default_q.copy()                               # 目标角=默认站姿
        q_des[self.active_idx] += action * cfg.action_scale        # +动作缩放
        kp = cfg.kp * kp_ratio                                     # 软启动缩放增益
        kd = cfg.kd * kp_ratio
        for _ in range(cfg.frame_skip):                            # PD+物理步×N
            pd_step(self.model, self.data, self.addr, q_des, kp, kd)
        self.prev_action = action                                  # 供下帧观测
