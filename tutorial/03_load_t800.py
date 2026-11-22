"""第 3 课 · 加载 T800 机器人。

把众擎官方 SDK 里的 t800.xml 加载进 MuJoCo，打印身体/关节/电机数量。
对应视频 P03 的「02/03 号程序」。

运行：python tutorial/03_load_t800.py
"""
import pathlib

import mujoco

# 本文件在 tutorial/ 下，上一级是项目根目录。
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "engineai_robotics_native_sdk" / "assets" / "resource" / "t800.xml"

# 25 个关节，顺序与模型文件 t800.xml 一致（J00..J24）。
JOINT_NAMES = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
    "J23_HEAD_PITCH", "J24_HEAD_YAW",
]


def load_model():
    """加载 T800 模型并复位（绝对路径，MuJoCo 4.0 下避免相对路径加载失败）。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def main():
    model, data = load_model()

    print(f"模型文件：{MODEL_PATH}")
    print(f"身体部件 nbody = {model.nbody}")
    print(f"关节数量 njnt = {model.njnt}（含 1 个浮基自由关节 + 25 个转动关节）")
    print(f"电机数量 nu    = {model.nu}")
    print(f"广义位置 nq    = {model.nq}（7 个浮基位姿 + 25 个关节角 = 32）")
    print(f"广义速度 nv    = {model.nv}（6 个浮基速度 + 25 个关节速度 = 31）")

    print("\n25 个关节：")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:2d}  {name}")


if __name__ == "__main__":
    main()
