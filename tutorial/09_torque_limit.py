"""第 9 课 · 力矩范围（control range）。

打印 25 个电机的力矩上限，演示用 np.clip 把超限力矩裁剪到安全范围。
对应视频 P09 的「13/14 号程序」。

运行：python tutorial/09_torque_limit.py
"""
import pathlib

import mujoco
import numpy as np

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


def build_addresses(model):
    """预计算 25 个关节的执行器地址与力矩上限，避免逐步字符串查找。"""
    aid = np.array([model.actuator("motor_" + n).id for n in JOINT_NAMES], dtype=int)
    tau_lim = model.actuator_ctrlrange[aid][:, 1].copy()  # 对称范围，取上限
    return {"aid": aid, "tau_lim": tau_lim}


def main():
    model, data = load_model()
    addr = build_addresses(model)

    print(f"{'ID':>2} {'关节':22s} {'ctrlrange':>18s}")
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = model.actuator_ctrlrange[addr["aid"][i]]
        print(f"{i:2d} {name:22s} [{lo:>6.0f}, {hi:>6.0f}]")

    # 演示裁剪：假设 PD 算出 5000 N·m（明显超限），裁剪到该电机的上限
    tau_raw = np.full(25, 5000.0)
    tau_safe = np.clip(tau_raw, -addr["tau_lim"], addr["tau_lim"])
    print("\n未裁剪 tau（前 3 项）：", tau_raw[:3])
    print("裁剪后 tau（前 3 项）：", tau_safe[:3], "（被压到各电机 ctrlrange 上限）")


if __name__ == "__main__":
    main()
