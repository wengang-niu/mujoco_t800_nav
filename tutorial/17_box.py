"""第 17 课 · 在世界里构建一个木块。

在 MuJoCo 世界里加一个带自由关节的红色木块（box），跑仿真看它落到地面。
对应视频 P17 的「24 号程序」。

运行：python tutorial/17_box.py
"""
import mujoco
import numpy as np

import common as C


def main():
    model, data = C.load_model_with_box(half=0.1, mass=5.0, pos=(1.5, 0.0, 0.8))
    print(f"新增木块后 body 数量 = {model.nbody}")

    for _ in range(300):                 # 跑 0.6s，看木块自由落体
        mujoco.mj_step(model, data)

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    print(f"木块落点 xyz = {np.round(data.xpos[box_id], 3)}")
    print("（把木块加到模型后，后续课程让 T800 踢它/拿它）")


if __name__ == "__main__":
    main()
