"""第 12 课 · T800 的观测空间（1083 的来历）。

拆解 1083 = 15 帧历史 × 72 维单帧 + 3 维指令。打印单帧 72 维各段的
顺序与取值。对应视频 P12。

运行：python tutorial/12_observation.py
"""
import numpy as np

import common as C


def main():
    w = C.Walker()
    obs = w._obs()   # 单帧 72 维

    # 单帧 72 维 = 关节位置偏差 22 + 关节速度 22 + 上一动作 22 + 角速度 3 + 投影重力 3
    pos_err = obs[0:22]
    vel = obs[22:44]
    prev = obs[44:66]
    ang = obs[66:69]
    grav = obs[69:72]

    print("单帧 72 维观测的各段：")
    print(f"  [0:22]   关节位置偏差（q - default_q）× 缩放 1.0  = {np.round(pos_err[:3], 3)} ...")
    print(f"  [22:44]  关节速度（qd）× 缩放 0.05                = {np.round(vel[:3], 3)} ...")
    print(f"  [44:66]  上一动作 prev_action                      = {np.round(prev[:3], 3)} ...")
    print(f"  [66:69]  身体角速度（body 系）                     = {np.round(ang, 3)}")
    print(f"  [69:72]  投影重力（直立≈[0,0,-1]）                 = {np.round(grav, 3)}")

    # 历史 15 帧 + 指令 3 维 = 1083
    full = np.concatenate(w.history + [np.array([0.2, 0, 0]) * C.CMD_SCALE])
    print(f"\n历史 {C.HISTORY} 帧 × 72 = {C.HISTORY * 72}，+ 指令 3 = {full.shape[0]} 维")


if __name__ == "__main__":
    main()
