"""第 10 课 · 手动控制 T800（键盘 WASD 改目标角）。

用 MuJoCo viewer 的按键回调，把按键映射为某个关节目标角的增减，经 PD 跟踪，
实时观察关节运动。演示「手动 → 单关节」的控制链路（对应视频 P10 的思路）。
按键：W/S 增大/减小目标角，Q 退出。

运行：python tutorial/10_manual.py
"""
import mujoco
import mujoco.viewer  # 需显式导入，否则 mujoco.viewer 不存在
import numpy as np

import common as C

TARGET_JOINT = "J03_KNEE_PITCH_L"   # 演示用关节
STEP = 0.05                          # 每次按键改变的目标角（rad）


def main():
    model, data = C.load_model()
    addr = C.build_addresses(model)
    jid = C.JOINT_NAMES.index(TARGET_JOINT)

    data.qpos[2] = C.BASE_Z
    data.qpos[addr["qadr"]] = C.DEFAULT_Q
    mujoco.mj_forward(model, data)

    target = C.DEFAULT_Q.copy()

    def key_cb(keycode: int):
        nonlocal target
        if keycode == 87:        # W
            target[jid] += STEP
            print(f"目标角 {TARGET_JOINT} -> {target[jid]:+.3f} rad")
        elif keycode == 83:      # S
            target[jid] -= STEP
            print(f"目标角 {TARGET_JOINT} -> {target[jid]:+.3f} rad")
        elif keycode == 81:      # Q
            raise SystemExit

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            C.pd_step(model, data, addr, target)
            viewer.sync()


if __name__ == "__main__":
    main()
