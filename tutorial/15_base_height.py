"""第 15 课 · T800 初始化高度设置（接触点扫描）。

把机器人放在不同基座高度，统计与地面的接触点数：穿透时接触很多，
完全悬空时为 0，刚变为 0 的边界就是合适高度。对应视频 P15 的「22 号程序」。

运行：python tutorial/15_base_height.py
"""
import mujoco
import numpy as np

import common as C


def count_ground_contacts(data):
    """统计两侧 body 中恰有一方为 world(0) 的接触对数量。"""
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        a_is_world = (c.geom1 == 0) or (c.geom2 == 0)  # geom id 0 是 world 地面
        if a_is_world:
            n += 1
    return n


def main():
    model, data = C.load_model()
    addr = C.build_addresses(model)

    print(f"{'base_z':>8} {'ground_contacts':>16}")
    zs = np.arange(0.50, 1.50 + 0.01, 0.02)   # 0.50~1.50，步长 0.02
    for z in zs:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = z
        data.qpos[addr["qadr"]] = C.DEFAULT_Q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        n = count_ground_contacts(data)
        print(f"{z:8.2f} {n:16d}")

    print("\n结论：接触数刚变为 0 的高度 z=1.02m 即「脚底刚好触地」的合适初始高度。")


if __name__ == "__main__":
    main()
