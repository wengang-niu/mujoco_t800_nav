"""第 17 课 · 在世界里构建一个木块。

在 MuJoCo 世界里加一个带自由关节的红色木块（box），跑仿真看它落到地面。
对应视频 P17 的「24 号程序」。

运行：python tutorial/17_box.py
"""
import pathlib

import mujoco
import numpy as np

# 本文件在 tutorial/ 下，上一级是项目根目录。
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "engineai_robotics_native_sdk" / "assets" / "resource" / "t800.xml"


def load_model_with_box(half=0.1, mass=1.0, pos=(1.2, 0.0, 0.1), collidable=True):
    """加载 T800 模型并追加一个带自由关节的木块（用 MjSpec 编译，保留网格资源）。

    collidable=False 时木块不参与碰撞（contype/conaffinity=0），用于「坐标绑定」
    这种把木块贴到手部的展示场景，避免木块与手部碰撞干扰行走。
    """
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    body = spec.worldbody.add_body()
    body.name = "box"
    body.pos = list(pos)
    body.add_freejoint()
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [half, half, half]
    geom.mass = mass
    geom.rgba = [0.8, 0.2, 0.1, 1.0]
    if not collidable:
        geom.contype = 0
        geom.conaffinity = 0
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def main():
    model, data = load_model_with_box(half=0.1, mass=5.0, pos=(1.5, 0.0, 0.8))
    print(f"新增木块后 body 数量 = {model.nbody}")

    for _ in range(300):                 # 跑 0.6s，看木块自由落体
        mujoco.mj_step(model, data)

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    print(f"木块落点 xyz = {np.round(data.xpos[box_id], 3)}")
    print("（把木块加到模型后，后续课程让 T800 踢它/拿它）")


if __name__ == "__main__":
    main()
