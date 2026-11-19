"""第 8 课 · PD 控制。

PD 控制器把关节拉到目标角：tau = kp*(q_des - q) + kd*(0 - qd)。
本课用官方 kp/kd 把 25 个关节锁在默认姿态，观察机器人能否保持姿态。
对应视频 P08 的「12 号程序」。

运行：python tutorial/08_pd.py
"""
import mujoco
import numpy as np

import common as C


def main():
    model, data = C.load_model()
    addr = C.build_addresses(model)

    # 初始：默认姿态 + 脚底触地高度
    data.qpos[2] = C.BASE_Z
    data.qpos[addr["qadr"]] = C.DEFAULT_Q
    mujoco.mj_forward(model, data)

    print(f"{'t(s)':>6} {'base_z':>8} {'pitch_deg':>9} {'max|err|':>9}")
    total = int(3.0 / model.opt.timestep)
    for step in range(total):
        tau = C.pd_step(model, data, addr, C.DEFAULT_Q)   # 目标 = 默认姿态
        if step % 50 == 0:
            proj_g = C.projected_gravity(data)
            pitch = np.rad2deg(np.arctan2(proj_g[0], -proj_g[2]))
            err = np.max(np.abs(data.qpos[addr["qadr"]] - C.DEFAULT_Q))
            print(f"{data.time:6.2f} {data.qpos[2]:8.3f} {pitch:9.2f} {err:9.4f}")
    print("结论：纯固定 PD 只能让关节跟踪目标，浮基双足仍会缓慢前倾——"
          "真正站稳需要策略实时调整目标角（后续课程）。")


if __name__ == "__main__":
    main()
