"""第 20 课 · 调试任务拆解（最小原子任务）。

把「边走路边抓取」拆成互不干扰的子任务。本课演示其中最小的一块：
固定 base（锁住浮基），只让右臂在「放下 / 抬起」两个目标姿态间用
alpha 插值 + smoothstep 平滑过渡。对应视频 P20 的「27 号程序」。

运行：python tutorial/20_task_decompose.py
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

# 右臂 5 个关节的抬臂目标偏移（相对默认姿态）
RAISE_DELTA = {
    "J18_SHOULDER_PITCH_R": -1.0,
    "J19_SHOULDER_ROLL_R": -0.35,
    "J20_SHOULDER_YAW_R": 0.10,
    "J21_ELBOW_PITCH_R": -0.45,
    "J22_ELBOW_YAW_R": 0.00,
}
ARM_NAMES = list(RAISE_DELTA.keys())
ARM_IDX = [JOINT_NAMES.index(n) for n in ARM_NAMES]
BLEND_TIME = 0.8   # 插补时长（s）


def load_model():
    """加载 T800 模型并复位（绝对路径，MuJoCo 4.0 下避免相对路径加载失败）。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def build_addresses(model):
    """预计算 25 个关节的 qpos/qvel/执行器地址与力矩上限，避免逐步字符串查找。"""
    qadr = np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES], dtype=int)
    dadr = np.array([model.joint(n).dofadr[0] for n in JOINT_NAMES], dtype=int)
    aid = np.array([model.actuator("motor_" + n).id for n in JOINT_NAMES], dtype=int)
    tau_lim = model.actuator_ctrlrange[aid][:, 1].copy()
    return {"qadr": qadr, "dadr": dadr, "aid": aid, "tau_lim": tau_lim}


def pd_step(model, data, addr, q_des, kp=KP, kd=KD):
    """全身 PD：tau = kp*(q_des-q) + kd*(0-qd)，按力矩限幅后写入 ctrl 并推进一步。"""
    q = data.qpos[addr["qadr"]]
    qd = data.qvel[addr["dadr"]]
    tau = kp * (q_des - q) + kd * (0.0 - qd)
    tau = np.clip(tau, -addr["tau_lim"], addr["tau_lim"])
    data.ctrl[addr["aid"]] = tau
    mujoco.mj_step(model, data)
    return tau


def smoothstep(x):
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return x * x * (3.0 - 2.0 * x)


def main():
    model, data = load_model()
    addr = build_addresses(model)
    data.qpos[2] = BASE_Z
    data.qpos[addr["qadr"]] = DEFAULT_Q
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos[:7].copy()   # 固定 base 的位姿

    q_lower = DEFAULT_Q.copy()       # 放下（默认姿态）
    q_raise = DEFAULT_Q.copy()       # 抬起
    for i, n in enumerate(ARM_NAMES):
        q_raise[ARM_IDX[i]] += RAISE_DELTA[n]

    total = int(4.0 / model.opt.timestep)
    for step in range(total):
        t = data.time
        # 0~0.8s 抬起，1.6~2.4s 放下，其余保持
        if t < BLEND_TIME:
            alpha = smoothstep(t / BLEND_TIME)
            q_des = q_lower + alpha * (q_raise - q_lower)
        elif 1.6 <= t < 1.6 + BLEND_TIME:
            alpha = smoothstep((t - 1.6) / BLEND_TIME)
            q_des = q_raise + alpha * (q_lower - q_raise)
        elif t < 1.6:
            q_des = q_raise
        else:
            q_des = q_lower

        pd_step(model, data, addr, q_des)
        # 固定 base：每步把浮基位姿/速度钉回初始值（只让手臂动）
        data.qpos[:7] = base_qpos
        data.qvel[:6] = 0.0
        if step % 200 == 0:
            elbow = data.qpos[addr["qadr"][JOINT_NAMES.index("J21_ELBOW_PITCH_R")]]
            print(f"t={t:4.2f}s  右肘角 J21={elbow:+.3f} rad（base 固定，仅手臂在动）")
    print("结束：固定 base、只动右臂的最小原子任务跑通。")


if __name__ == "__main__":
    main()
