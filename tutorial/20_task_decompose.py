"""第 20 课 · 调试任务拆解（最小原子任务）。

把「边走路边抓取」拆成互不干扰的子任务。本课演示其中最小的一块：
固定 base（锁住浮基），只让右臂在「放下 / 抬起」两个目标姿态间用
alpha 插值 + smoothstep 平滑过渡。对应视频 P20 的「27 号程序」。

运行：python tutorial/20_task_decompose.py
"""
import mujoco
import numpy as np

import common as C

# 右臂 5 个关节的抬臂目标偏移（相对默认姿态）
RAISE_DELTA = {
    "J18_SHOULDER_PITCH_R": -1.0,
    "J19_SHOULDER_ROLL_R": -0.35,
    "J20_SHOULDER_YAW_R": 0.10,
    "J21_ELBOW_PITCH_R": -0.45,
    "J22_ELBOW_YAW_R": 0.00,
}
ARM_NAMES = list(RAISE_DELTA.keys())
ARM_IDX = [C.JOINT_NAMES.index(n) for n in ARM_NAMES]
BLEND_TIME = 0.8   # 插补时长（s）


def smoothstep(x):
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return x * x * (3.0 - 2.0 * x)


def main():
    model, data = C.load_model()
    addr = C.build_addresses(model)
    data.qpos[2] = C.BASE_Z
    data.qpos[addr["qadr"]] = C.DEFAULT_Q
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos[:7].copy()   # 固定 base 的位姿

    q_lower = C.DEFAULT_Q.copy()       # 放下（默认姿态）
    q_raise = C.DEFAULT_Q.copy()       # 抬起
    for i, n in enumerate(ARM_NAMES):
        q_raise[ARM_IDX[i]] += RAISE_DELTA[n]

    total = int(4.0 / model.opt.timestep)
    for step in range(total):
        t = data.time
        # 0~0.8s 抬起，1.6~2.4s 放下，其余保持
        if t < BLEND_TIME:
            alpha = smoothstep(t / BLEND_TIME)
            q_des = q_lower + alpha * (q_raise - q_lower)
        elif 1.6 <= t < 1.6 + BLEND_TIME:
            alpha = smoothstep((t - 1.6) / BLEND_TIME)
            q_des = q_raise + alpha * (q_lower - q_raise)
        elif t < 1.6:
            q_des = q_raise
        else:
            q_des = q_lower

        C.pd_step(model, data, addr, q_des)
        # 固定 base：每步把浮基位姿/速度钉回初始值（只让手臂动）
        data.qpos[:7] = base_qpos
        data.qvel[:6] = 0.0
        if step % 200 == 0:
            elbow = data.qpos[addr["qadr"][C.JOINT_NAMES.index("J21_ELBOW_PITCH_R")]]
            print(f"t={t:4.2f}s  右肘角 J21={elbow:+.3f} rad（base 固定，仅手臂在动）")
    print("结束：固定 base、只动右臂的最小原子任务跑通。")


if __name__ == "__main__":
    main()
