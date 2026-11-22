"""第 8 课 · PD 控制。

PD 控制器把关节拉到目标角：tau = kp*(q_des - q) + kd*(0 - qd)。
本课用官方 kp/kd 把 25 个关节锁在默认姿态，观察机器人能否保持姿态。
对应视频 P08 的「12 号程序」。

运行：python tutorial/08_pd.py
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


def quat_to_rot(q):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵（body -> world）。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def projected_gravity(data):
    """世界重力 [0,0,-1] 投影到基座坐标系（直立时 ≈ [0,0,-1]）。"""
    R = quat_to_rot(data.qpos[3:7])
    return R.T @ np.array([0.0, 0.0, -1.0])


def main():
    model, data = load_model()
    addr = build_addresses(model)

    # 初始：默认姿态 + 脚底触地高度
    data.qpos[2] = BASE_Z
    data.qpos[addr["qadr"]] = DEFAULT_Q
    mujoco.mj_forward(model, data)

    print(f"{'t(s)':>6} {'base_z':>8} {'pitch_deg':>9} {'max|err|':>9}")
    total = int(3.0 / model.opt.timestep)
    for step in range(total):
        tau = pd_step(model, data, addr, DEFAULT_Q)   # 目标 = 默认姿态
        if step % 50 == 0:
            proj_g = projected_gravity(data)
            pitch = np.rad2deg(np.arctan2(proj_g[0], -proj_g[2]))
            err = np.max(np.abs(data.qpos[addr["qadr"]] - DEFAULT_Q))
            print(f"{data.time:6.2f} {data.qpos[2]:8.3f} {pitch:9.2f} {err:9.4f}")
    print("结论：纯固定 PD 只能让关节跟踪目标，浮基双足仍会缓慢前倾——"
          "真正站稳需要策略实时调整目标角（后续课程）。")


if __name__ == "__main__":
    main()
