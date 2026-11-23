"""第 7 课 · MuJoCo 仿真主循环。

mj_step 是推进仿真「一帧」的核心调用。本课演示「读取状态 → 写入控制 → mj_step」
三部曲：无控制时机器人因重力倒地，用 3D 窗口直接观看。
对应视频 P07 的「11 号程序」。

运行：
  python tutorial/07_loop.py     # 打开 viewer 观看
"""
import pathlib
import time

import mujoco
import mujoco.viewer  # 需显式导入，否则 mujoco.viewer 不存在
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

# 官方默认姿态：各关节相对零位（直腿）的偏移，单位 rad。
DEFAULT_Q = np.array(
    [-0.06, 0.0, 0.0, 0.12, -0.06, 0.0,     # 左腿
     -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,     # 右腿
     0.0,                                    # 腰
     0.0, 0.15, 0.0, -0.25, 0.0,             # 左臂
     0.0, -0.15, 0.0, -0.25, 0.0,            # 右臂
     0.0, 0.0],                              # 头
    dtype=np.float64,
)
BASE_Z = 1.02   # 初始化基座高度（接触扫描得出的「脚底刚触地」高度）


def load_model():
    """加载 T800 模型并复位（绝对路径，MuJoCo 4.0 下避免相对路径加载失败）。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def main():
    model, data = load_model()

    # —— 初始化姿态：把机器人摆成官方默认站姿 ——
    data.qpos[2] = BASE_Z                      # 基座 Z 抬到「脚底刚触地」的高度
    for i, n in enumerate(JOINT_NAMES):
        data.joint(n).qpos[0] = DEFAULT_Q[i]   # 访问器写入：等价于向 data.qpos 写该关节角
    mujoco.mj_forward(model, data)             # 用新状态重算前向运动学

    run_viewer(model, data)


def run_viewer(model, data):
    """打开 3D 窗口，循环推进仿真，实时观看机器人因重力倒地。"""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[:] = 0.0                 # 无控制：所有电机不给力矩
            mujoco.mj_step(model, data)        # 推进一帧
            viewer.sync()                      # 把最新状态同步到渲染窗口
            time.sleep(model.opt.timestep)     # 按真实时间节奏跑，别刷太快


if __name__ == "__main__":
    main()
