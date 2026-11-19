"""第 7 课 · MuJoCo 仿真主循环。

mj_step 是推进仿真「一帧」的核心调用。本课演示「读取状态 → 写入控制 → mj_step」
三部曲：无控制时机器人因重力倒地，并打印基座高度/倾角。
对应视频 P07 的「11 号程序」。

运行：
  python tutorial/07_loop.py              # 无窗口跑 3 秒，打印诊断
  python tutorial/07_loop.py --viewer     # 打开 viewer 观看
"""
import argparse
import time

import mujoco
import mujoco.viewer  # 需显式导入，否则 mujoco.viewer 不存在
import numpy as np

import common as C


def main():
    args = parse_args()
    model, data = C.load_model()
    # 初始姿态设为默认姿态、基座抬到脚底触地高度
    data.qpos[2] = C.BASE_Z
    for i, n in enumerate(C.JOINT_NAMES):
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] = C.DEFAULT_Q[i]
    mujoco.mj_forward(model, data)

    if args.viewer:
        run_viewer(model, data)
    else:
        run_headless(model, data)


def run_headless(model, data, seconds=3.0):
    addr = C.build_addresses(model)
    total = int(seconds / model.opt.timestep)
    print(f"{'t(s)':>6} {'base_z':>8} {'pitch_deg':>9}")
    for step in range(total):
        data.ctrl[:] = 0.0            # 无控制：所有电机不给力矩
        mujoco.mj_step(model, data)   # 推进一帧
        if step % 50 == 0:
            proj_g = C.projected_gravity(data)
            pitch = np.rad2deg(np.arctan2(proj_g[0], -proj_g[2]))
            print(f"{data.time:6.2f} {data.qpos[2]:8.3f} {pitch:9.2f}")
    print("结论：无电机力矩时，机器人因重力倒下（这是浮基双足的正常表现）。")


def run_viewer(model, data):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--viewer", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
