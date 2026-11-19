"""第 16 课 · 运动控制二次开发（上下左右键遥控行走）。

在完整行走闭环上，用 viewer 按键回调实时改写三维指令 command=[vx, vy, yaw_rate]，
实现 ↑/↓ 前进后退、←/→ 左右转向。对应视频 P16 的「23 号程序」。

运行：python tutorial/16_wasd.py
"""
import time

import mujoco
import mujoco.viewer  # 需显式导入，否则 mujoco.viewer 不存在

import common as C

# GLFW 键码（MuJoCo viewer 的 key_callback 传入的即 GLFW 键码）
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_SPACE = 262, 263, 264, 265, 32


def main():
    w = C.Walker()
    cmd = [0.0, 0.0, 0.0]
    step = 0.2            # 每次按键改变的速度（m/s 或 rad/s）
    transition = int(0.5 / C.CONTROL_DT)

    def key_cb(keycode: int):
        nonlocal cmd
        if keycode == KEY_UP:      # ↑ 前进
            cmd[0] = min(cmd[0] + step, 0.6)
        elif keycode == KEY_DOWN:  # ↓ 后退
            cmd[0] = max(cmd[0] - step, -0.6)
        elif keycode == KEY_LEFT:  # ← 左转
            cmd[2] = min(cmd[2] + step, 1.0)
        elif keycode == KEY_RIGHT:  # → 右转
            cmd[2] = max(cmd[2] - step, -1.0)
        elif keycode == KEY_SPACE:  # 空格停止
            cmd = [0.0, 0.0, 0.0]
        print(f"command = [{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}]")

    with mujoco.viewer.launch_passive(w.model, w.data, key_callback=key_cb) as viewer:
        k = 0
        while viewer.is_running():
            ratio = min(1.0, k / transition)
            w.step(command=cmd, kp_ratio=0.2 + 0.8 * ratio)
            viewer.sync()
            time.sleep(C.CONTROL_DT)
            k += 1


if __name__ == "__main__":
    main()
