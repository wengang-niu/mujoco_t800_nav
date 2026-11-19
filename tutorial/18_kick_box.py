"""第 18 课 · T800 踢着木块往前走。

在机器人前方放一个木块，跑行走策略前进，观察机器人把木块推着/踢着向前。
对应视频 P18 的「25 号程序」。

运行：python tutorial/18_kick_box.py
"""
import mujoco

import common as C


def main():
    model, data = C.load_model_with_box(half=0.1, mass=1.0, pos=(1.2, 0.0, 0.1))
    w = C.Walker(model, data)
    box_id = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "box")

    transition = int(0.5 / C.CONTROL_DT)
    total = int(8.0 / C.CONTROL_DT)
    print(f"{'t(s)':>6} {'robot_x':>8} {'box_x':>8}")
    for k in range(total):
        ratio = min(1.0, k / transition)
        w.step(command=(0.2, 0.0, 0.0), kp_ratio=0.2 + 0.8 * ratio)
        if k % 100 == 0:
            print(f"{k * C.CONTROL_DT:6.2f} {w.data.qpos[0]:8.3f} "
                  f"{w.data.xpos[box_id][0]:8.3f}")
        if w.data.qpos[2] < 0.55:
            print("机器人摔倒"); return
    print(f"结束：机器人 x={w.data.qpos[0]:.2f}m，木块 x={w.data.xpos[box_id][0]:.2f}m（被推动前进）")


if __name__ == "__main__":
    main()
