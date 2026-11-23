"""第 5 课 · T800 的相对坐标与绝对坐标。

对比「局部坐标」model.body_pos（相对父 body）与「世界坐标」data.xpos
（正向运动学后相对世界）。对应视频 P05 的「07/08 号程序」。

运行：python tutorial/05_coordinates.py
"""
import pathlib

import mujoco
import numpy as np

# 本文件在 tutorial/ 下，上一级是项目根目录。
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "engineai_robotics_native_sdk" / "assets" / "resource" / "t800.xml"


def load_model():
    """加载 T800 模型并复位（绝对路径，MuJoCo 4.0 下避免相对路径加载失败）。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def quat_to_rot(q):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵（body -> world）。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main():
    model, data = load_model()
    mujoco.mj_forward(model, data)

    print(f"{'body':20s} {'局部 pos (相对父body)':30s} {'世界 xpos':28s}")
    for name in ["world", "LINK_BASE", "LINK_HIP_PITCH_L", "LINK_KNEE_PITCH_L",
                 "LINK_ANKLE_ROLL_L", "LINK_FOOT_L", "LINK_WRIST_END_R"]:
        bid = model.body(name).id
        lpos = model.body_pos[bid]          # 相对父 body 的模型定义位置
        wpos = data.xpos[bid]               # 世界坐标
        print(f"{name:20s} {lpos!s:30s} {wpos!s:28s}")


if __name__ == "__main__":
    main()
