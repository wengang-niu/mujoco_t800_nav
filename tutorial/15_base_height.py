"""第 15 课 · T800 初始化高度设置（接触点扫描）。

把机器人放在不同基座高度，统计与地面的接触点数：穿透时接触很多，
完全悬空时为 0，刚变为 0 的边界就是合适高度。对应视频 P15 的「22 号程序」。

运行：python tutorial/15_base_height.py
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

# 官方默认姿态（default.yaml）。
DEFAULT_Q = np.array(
    [-0.06, 0.0, 0.0, 0.12, -0.06, 0.0,     # 左腿
     -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,     # 右腿
     0.0,                                    # 腰
     0.0, 0.15, 0.0, -0.25, 0.0,             # 左臂
     0.0, -0.15, 0.0, -0.25, 0.0,            # 右臂
     0.0, 0.0],                              # 头
    dtype=np.float64,
)


def load_model():
    """加载 T800 模型并复位（绝对路径，MuJoCo 4.0 下避免相对路径加载失败）。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def build_addresses(model):
    """预计算 25 个关节的 qpos/qvel/执行器地址，避免逐步字符串查找。"""
    qadr = np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES], dtype=int)
    dadr = np.array([model.joint(n).dofadr[0] for n in JOINT_NAMES], dtype=int)
    return {"qadr": qadr, "dadr": dadr}


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
    model, data = load_model()
    addr = build_addresses(model)

    print(f"{'base_z':>8} {'ground_contacts':>16}")
    zs = np.arange(0.50, 1.50 + 0.01, 0.02)   # 0.50~1.50，步长 0.02
    for z in zs:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = z
        data.qpos[addr["qadr"]] = DEFAULT_Q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        n = count_ground_contacts(data)
        print(f"{z:8.2f} {n:16d}")

    print("\n结论：接触数刚变为 0 的高度 z=1.02m 即「脚底刚好触地」的合适初始高度。")


if __name__ == "__main__":
    main()
