"""第 8 课 · PD 控制。

PD 控制器把关节拉到目标角：tau = kp*(q_des - q) + kd*(0 - qd)。
本课用官方 kp/kd 把 25 个关节锁在默认姿态，在 3D 窗口里观察机器人会先站住、再缓慢前倾。
对应视频 P08 的「12 号程序」。

运行：python tutorial/08_pd.py
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

# 官方默认姿态与 PD 增益（default.yaml）。
DEFAULT_Q = np.array(
    [-0.06, 0.0, 0.0, 0.12, -0.06, 0.0,     # 左腿
     -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,     # 右腿
     0.0,                                    # 腰
     0.0, 0.15, 0.0, -0.25, 0.0,             # 左臂
     0.0, -0.15, 0.0, -0.25, 0.0,            # 右臂
     0.0, 0.0],                              # 头
    dtype=np.float64,
)
# P 增益（刚度）：把位置误差放大成力矩，越大越"硬"；顺序同 JOINT_NAMES。
KP = np.array(
    [180, 100, 100, 180, 40, 40,   180, 100, 100, 180, 40, 40,   100,
     60, 50, 50, 60, 50,   60, 50, 50, 60, 50,   100, 100], dtype=np.float64)
# D 增益（阻尼）：把速度误差放大成反向力矩，抑制过冲/振荡；顺序同 JOINT_NAMES。
KD = np.array(
    [5.0, 3.0, 3.0, 5.0, 0.3, 0.3,   5.0, 3.0, 3.0, 5.0, 0.3, 0.3,   5.0,
     0.3, 0.3, 0.3, 0.3, 0.3,   0.3, 0.3, 0.3, 0.3, 0.3,   1.0, 1.0], dtype=np.float64)
BASE_Z = 1.02   # 初始化基座高度（「脚底刚触地」）


def load_model():
    """加载 T800 模型并复位（绝对路径，MuJoCo 4.0 下避免相对路径加载失败）。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))  # 模型定义（图纸，静态）
    data = mujoco.MjData(model)                            # 运行时状态容器（此刻，动态）
    mujoco.mj_resetData(model, data)                       # 复位到零位姿态
    mujoco.mj_forward(model, data)                         # 前向运动学：算齐所有派生量
    return model, data


def build_addresses(model):
    """预计算 25 个关节的 qpos/qvel/执行器地址与力矩上限，避免逐步字符串查找。"""
    qadr = np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES], dtype=int)      # 关节角在 data.qpos 的下标
    dadr = np.array([model.joint(n).dofadr[0] for n in JOINT_NAMES], dtype=int)       # 关节速度在 data.qvel 的下标
    aid = np.array([model.actuator("motor_" + n).id for n in JOINT_NAMES], dtype=int)  # 关节对应电机的执行器 id
    tau_lim = model.actuator_ctrlrange[aid][:, 1].copy()                               # 各电机力矩上限（ctrlrange 第 2 列）
    return {"qadr": qadr, "dadr": dadr, "aid": aid, "tau_lim": tau_lim}


def pd_step(model, data, addr, q_des, kp=KP, kd=KD):
    """全身 PD：tau = kp*(q_des-q) + kd*(0-qd)，按力矩限幅后写入 ctrl 并推进一步。"""
    q = data.qpos[addr["qadr"]]                          # 当前关节角（一次性取出 25 个）
    qd = data.qvel[addr["dadr"]]                         # 当前关节角速度
    tau = kp * (q_des - q) + kd * (0.0 - qd)             # P 项拉回目标 + D 项阻尼（目标速度=0）
    tau = np.clip(tau, -addr["tau_lim"], addr["tau_lim"])  # 限幅到电机力矩上限，防过载
    data.ctrl[addr["aid"]] = tau                         # 按执行器 id 写入控制输入
    mujoco.mj_step(model, data)                          # 推进一帧仿真
    return tau


def main():
    model, data = load_model()
    addr = build_addresses(model)

    # 初始：默认姿态 + 脚底触地高度。
    data.qpos[2] = BASE_Z
    data.qpos[addr["qadr"]] = DEFAULT_Q
    mujoco.mj_forward(model, data)

    run_viewer(model, data, addr)


def run_viewer(model, data, addr):
    """打开 3D 窗口，实时观看全身 PD 把关节拉回默认姿态。"""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            pd_step(model, data, addr, DEFAULT_Q)  # 目标 = 默认姿态
            viewer.sync()                          # 把最新状态同步到渲染窗口
            time.sleep(model.opt.timestep)         # 按真实时间节奏跑，别刷太快


if __name__ == "__main__":
    main()
