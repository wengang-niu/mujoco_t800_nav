"""第 10 课 · 手动控制 T800（键盘 WASD 改目标角）。

用 MuJoCo viewer 的按键回调，把按键映射为某个关节目标角的增减，经 PD 跟踪，
实时观察关节运动。演示「手动 → 单关节」的控制链路（对应视频 P10 的思路）。
按键：W/S 增大/减小目标角，Q 退出。

运行：python tutorial/10_manual.py
"""
import pathlib

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
KP = np.array(
    [180, 100, 100, 180, 40, 40,   180, 100, 100, 180, 40, 40,   100,
     60, 50, 50, 60, 50,   60, 50, 50, 60, 50,   100, 100], dtype=np.float64)
KD = np.array(
    [5.0, 3.0, 3.0, 5.0, 0.3, 0.3,   5.0, 3.0, 3.0, 5.0, 0.3, 0.3,   5.0,
     0.3, 0.3, 0.3, 0.3, 0.3,   0.3, 0.3, 0.3, 0.3, 0.3,   1.0, 1.0], dtype=np.float64)
BASE_Z = 1.02   # 初始化基座高度（「脚底刚触地」）

TARGET_JOINT = "J03_KNEE_PITCH_L"   # 演示用关节
STEP = 0.05                          # 每次按键改变的目标角（rad）


def load_model():
    """加载 T800 模型并复位（绝对路径，MuJoCo 4.0 下避免相对路径加载失败）。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def build_addresses(model):
    """预计算 25 个关节的 qpos/qvel/执行器地址与力矩上限，避免逐步字符串查找。"""
    jid = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in JOINT_NAMES], dtype=int)
    qadr = model.jnt_qposadr[jid].astype(int)
    dadr = model.jnt_dofadr[jid].astype(int)
    aid = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_" + n)
                    for n in JOINT_NAMES], dtype=int)
    tau_lim = model.actuator_ctrlrange[aid][:, 1].copy()
    return {"jid": jid, "qadr": qadr, "dadr": dadr, "aid": aid, "tau_lim": tau_lim}


def pd_step(model, data, addr, q_des, kp=KP, kd=KD):
    """全身 PD：tau = kp*(q_des-q) + kd*(0-qd)，按力矩限幅后写入 ctrl 并推进一步。"""
    q = data.qpos[addr["qadr"]]
    qd = data.qvel[addr["dadr"]]
    tau = kp * (q_des - q) + kd * (0.0 - qd)
    tau = np.clip(tau, -addr["tau_lim"], addr["tau_lim"])
    data.ctrl[addr["aid"]] = tau
    mujoco.mj_step(model, data)
    return tau


def main():
    model, data = load_model()
    addr = build_addresses(model)
    jid = JOINT_NAMES.index(TARGET_JOINT)

    data.qpos[2] = BASE_Z
    data.qpos[addr["qadr"]] = DEFAULT_Q
    mujoco.mj_forward(model, data)

    target = DEFAULT_Q.copy()

    def key_cb(keycode: int):
        nonlocal target
        if keycode == 87:        # W
            target[jid] += STEP
            print(f"目标角 {TARGET_JOINT} -> {target[jid]:+.3f} rad")
        elif keycode == 83:      # S
            target[jid] -= STEP
            print(f"目标角 {TARGET_JOINT} -> {target[jid]:+.3f} rad")
        elif keycode == 81:      # Q
            raise SystemExit

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            pd_step(model, data, addr, target)
            viewer.sync()


if __name__ == "__main__":
    main()
