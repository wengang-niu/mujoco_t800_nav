"""第 7 课 · MuJoCo 仿真主循环。

mj_step 是推进仿真「一帧」的核心调用。本课演示「读取状态 → 写入控制 → mj_step」
三部曲：无控制时机器人因重力倒地，并打印基座高度/倾角。
对应视频 P07 的「11 号程序」。

运行：
  python tutorial/07_loop.py              # 无窗口跑 3 秒，打印诊断
  python tutorial/07_loop.py --viewer     # 打开 viewer 观看
"""
import argparse
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
    args = parse_args()
    model, data = load_model()
    # 初始姿态设为默认姿态、基座抬到脚底触地高度
    data.qpos[2] = BASE_Z
    for i, n in enumerate(JOINT_NAMES):
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] = DEFAULT_Q[i]
    mujoco.mj_forward(model, data)

    if args.viewer:
        run_viewer(model, data)
    else:
        run_headless(model, data)


def run_headless(model, data, seconds=3.0):
    total = int(seconds / model.opt.timestep)
    print(f"{'t(s)':>6} {'base_z':>8} {'pitch_deg':>9}")
    for step in range(total):
        data.ctrl[:] = 0.0            # 无控制：所有电机不给力矩
        mujoco.mj_step(model, data)   # 推进一帧
        if step % 50 == 0:
            proj_g = projected_gravity(data)
            pitch = np.rad2deg(np.arctan2(proj_g[0], -proj_g[2]))
            print(f"{data.time:6.2f} {data.qpos[2]:8.3f} {pitch:9.2f}")
    print("结论：无电机力矩时，机器人因重力倒下（这是浮基双足的正常表现）。")


def run_viewer(model, data):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--viewer", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
