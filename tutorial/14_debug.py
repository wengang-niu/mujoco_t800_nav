"""第 14 课 · 仿真调试策略（日志为王 + 启动过渡）。

跑完整行走闭环，打印基座高度/倾角/位移与动作量级，观察数值是否发散，
并演示「启动过渡」（前 0.5s 把 kp/kd 从 20% 渐变到 100%）与观测/动作裁剪。
对应视频 P14。

运行：python tutorial/14_debug.py [--seconds 10]
"""
import argparse

import numpy as np

import common as C


def main():
    args = parse_args()
    w = C.Walker()

    transition_steps = int(0.5 / C.CONTROL_DT)   # 启动过渡 0.5s
    total = int(args.seconds / C.CONTROL_DT)

    print(f"{'t(s)':>6} {'base_x':>8} {'base_z':>8} {'pitch_deg':>9} {'max|action|':>11}")
    for k in range(total):
        ratio = min(1.0, k / transition_steps)         # 0 -> 1
        kp_ratio = 0.2 + 0.8 * ratio                   # 前 0.5s 低增益过渡
        res = w.step(command=(0.2, 0.0, 0.0), kp_ratio=kp_ratio)

        if k % 100 == 0:
            proj_g = C.projected_gravity(w.data)
            pitch = np.rad2deg(np.arctan2(proj_g[0], -proj_g[2]))
            print(f"{k * C.CONTROL_DT:6.2f} {w.data.qpos[0]:8.3f} {w.data.qpos[2]:8.3f} "
                  f"{pitch:9.2f} {np.max(np.abs(res['action'])):11.3f}")

        if w.data.qpos[2] < 0.55:                      # 摔倒判据
            print(f"摔倒于 t={k * C.CONTROL_DT:.2f}s")
            return
    print(f"结束：前进 {w.data.qpos[0]:.2f}m，基座高度 {w.data.qpos[2]:.3f}m（稳定）")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=10.0)
    return p.parse_args()


if __name__ == "__main__":
    main()
