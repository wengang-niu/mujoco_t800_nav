"""读取 config/*.yaml 并转成可直接使用的配置对象（数组一律 numpy）。"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "t800.yaml"


@dataclass(frozen=True)
class Config:
    # 路径（已解析为绝对 Path）
    model_path: pathlib.Path
    policy_path: pathlib.Path
    # 关节
    joint_names: list          # 25
    active_joint_names: list   # 22
    active_idx: np.ndarray     # (22,) int，主动关节在 25 中的下标
    # 控制参数
    default_q: np.ndarray      # (25,)
    kp: np.ndarray             # (25,)
    kd: np.ndarray             # (25,)
    action_scale: np.ndarray   # (22,)
    # 观测
    obs_scale: np.ndarray      # (72,) 按 22+22+22+3+3 组装
    obs_clip: float
    history: int
    # 指令
    cmd_scale: np.ndarray      # (3,)
    action_clip: float
    # 时序
    control_dt: float
    timestep: float
    frame_skip: int
    # 复位
    base_z: float
    base_pos: np.ndarray       # (2,) 出生点 x,y，默认原点
    # 遥控
    cmd_step: float
    vx_limit: float
    yaw_limit: float
    soft_start_duration: float
    soft_start_ratio_min: float


def load_config(path: pathlib.Path | None = None) -> Config:
    path = pathlib.Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    root = ROOT
    model_path = root / raw["paths"]["model"]
    policy_path = root / raw["paths"]["policy"]

    joint_names = list(raw["joints"]["names"])
    active_joint_names = list(raw["joints"]["active"])
    if len(joint_names) != 25:
        raise ValueError(f"joints.names 需要 25 个，实际 {len(joint_names)}")
    if len(active_joint_names) != 22:
        raise ValueError(f"joints.active 需要 22 个，实际 {len(active_joint_names)}")
    active_idx = np.array(
        [joint_names.index(n) for n in active_joint_names], dtype=int)

    ctl = raw["control"]
    default_q = np.asarray(ctl["default_q"], dtype=np.float64)
    kp = np.asarray(ctl["kp"], dtype=np.float64)
    kd = np.asarray(ctl["kd"], dtype=np.float64)
    action_scale = np.asarray(ctl["action_scale"], dtype=np.float64)
    for name, arr, n in (("default_q", default_q, 25), ("kp", kp, 25),
                         ("kd", kd, 25), ("action_scale", action_scale, 22)):
        if arr.shape != (n,):
            raise ValueError(f"control.{name} 需要 ({n},)，实际 {arr.shape}")

    obs = raw["observation"]
    # 72 维 = 22 角偏差 + 22 角速度 + 22 上一动作 + 3 角速度 + 3 重力投影
    obs_scale = np.concatenate([
        np.full(22, float(obs["joint_pos_scale"])),
        np.full(22, float(obs["joint_vel_scale"])),
        np.full(22, float(obs["prev_action_scale"])),
        np.full(3, float(obs["ang_vel_scale"])),
        np.full(3, float(obs["gravity_scale"])),
    ])
    assert obs_scale.shape == (72,), obs_scale.shape

    cmd = raw["command"]
    cmd_scale = np.asarray(cmd["scale"], dtype=np.float64)
    if cmd_scale.shape != (3,):
        raise ValueError(f"command.scale 需要 (3,)，实际 {cmd_scale.shape}")

    tim = raw["timing"]
    control_dt = float(tim["control_dt"])
    timestep = float(tim["timestep"])
    frame_skip = int(round(control_dt / timestep))

    tele = raw["teleop"]
    reset = raw["reset"]

    return Config(
        model_path=model_path,
        policy_path=policy_path,
        joint_names=joint_names,
        active_joint_names=active_joint_names,
        active_idx=active_idx,
        default_q=default_q,
        kp=kp,
        kd=kd,
        action_scale=action_scale,
        obs_scale=obs_scale,
        obs_clip=float(obs["clip"]),
        history=int(obs["history"]),
        cmd_scale=cmd_scale,
        action_clip=float(cmd["action_clip"]),
        control_dt=control_dt,
        timestep=timestep,
        frame_skip=frame_skip,
        base_z=float(reset["base_z"]),
        base_pos=np.asarray(reset.get("base_pos", (0.0, 0.0)), dtype=np.float64),
        cmd_step=float(tele["cmd_step"]),
        vx_limit=float(tele["vx_limit"]),
        yaw_limit=float(tele["yaw_limit"]),
        soft_start_duration=float(tele["soft_start_duration"]),
        soft_start_ratio_min=float(tele["soft_start_ratio_min"]),
    )
