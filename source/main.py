"""T800 遥控行走 demo 入口。

用 viewer 按键回调实时改写三维指令 command=[vx, vy, yaw_rate]，
实现 ↑/↓ 前进后退、←/→ 左右转向。

运行：
    python source/main.py   # 打开 viewer 遥控；qpos 同步写到 /tmp/t800_qpos.bin
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import mujoco
import mujoco.viewer  # 需显式导入，否则 mujoco.viewer 不存在
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import load_config            # noqa: E402
from walker import Walker                 # noqa: E402

STATE_PATH = pathlib.Path("/tmp/t800_qpos.bin")  # camera_view.py 读取

# GLFW 键码（MuJoCo viewer 的 key_callback 传入的即 GLFW 键码）
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_SPACE = 262, 263, 264, 265, 32


def main():
    parser = argparse.ArgumentParser(description="T800 遥控行走")
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="配置文件路径，默认 config/t800.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    w = Walker(cfg)

    cmd = [0.0, 0.0, 0.0]  # [vx 前后, vy 侧移(未绑定按键), yaw_rate 转向]
    step = cfg.cmd_step
    vx_lo, vx_hi = -cfg.vx_limit, cfg.vx_limit
    yaw_lo, yaw_hi = -cfg.yaw_limit, cfg.yaw_limit
    transition = int(cfg.soft_start_duration / cfg.control_dt)
    r_min = cfg.soft_start_ratio_min

    def key_cb(keycode: int):
        nonlocal cmd
        if keycode == KEY_UP:        # ↑ 前进
            cmd[0] = min(cmd[0] + step, vx_hi)
        elif keycode == KEY_DOWN:    # ↓ 后退
            cmd[0] = max(cmd[0] - step, vx_lo)
        elif keycode == KEY_LEFT:    # ← 左转
            cmd[2] = min(cmd[2] + step, yaw_hi)
        elif keycode == KEY_RIGHT:   # → 右转
            cmd[2] = max(cmd[2] - step, yaw_lo)
        elif keycode == KEY_SPACE:   # 空格停止
            cmd = [0.0, 0.0, 0.0]
        print(f"command = [{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}]")

    # 发布 qpos 供 camera_view.py 读取（复用旧文件，避免打断已在运行的读端）
    nbytes = w.model.nq * np.dtype(np.float64).itemsize
    reuse = STATE_PATH.exists() and STATE_PATH.stat().st_size == nbytes
    state = np.memmap(STATE_PATH, dtype=np.float64,
                      mode="r+" if reuse else "w+", shape=(w.model.nq,))

    with mujoco.viewer.launch_passive(w.model, w.data, key_callback=key_cb) as viewer:
        k = 0
        while viewer.is_running():
            # 软启动：开局前 soft_start_duration 内 PD 增益从 r_min 线性升到 100%，
            # 避免满增益瞬间把机器人"钉死"或弹倒
            ratio = min(1.0, k / transition)
            w.step(command=cmd, kp_ratio=r_min + (1.0 - r_min) * ratio)
            state[:] = w.data.qpos
            viewer.sync()
            time.sleep(cfg.control_dt)
            k += 1


if __name__ == "__main__":
    main()
