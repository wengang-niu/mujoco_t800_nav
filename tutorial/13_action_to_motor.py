"""第 13 课 · AI 指令到机器人肌肉运动（动作映射）。

网络输出 22 维归一化动作，不能直接当力矩。先乘 action_scale 得到目标角偏移，
再叠加默认姿态得到目标角，最后经 PD 换算成力矩。对应视频 P13。

运行：python tutorial/13_action_to_motor.py
"""
import pathlib

import mujoco
import numpy as np

# ---- 路径（本文件在 tutorial/ 下，上一级是项目根目录）----
ROOT = pathlib.Path(__file__).resolve().parent.parent
SDK = ROOT / "engineai_robotics_native_sdk"
MODEL_PATH = SDK / "assets" / "resource" / "t800.xml"
POLICY_PATH = (SDK / "assets" / "config" / "t800" / "rl_walking_example" / "policy"
               / "t800_260618_165257_30000.mnn")

# ---- 25 个关节，顺序与 t800.xml 一致（J00..J24）----
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
ACTIVE_JOINT_NAMES = JOINT_NAMES[:12] + JOINT_NAMES[13:23]

# ---- 官方 SDK 参数（default.yaml）----
DEFAULT_Q = np.array(
    [-0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
     -0.06, 0.0, 0.0, 0.12, -0.06, 0.0,
     0.0,
     0.0, 0.15, 0.0, -0.25, 0.0,
     0.0, -0.15, 0.0, -0.25, 0.0,
     0.0, 0.0], dtype=np.float64)
KP = np.array(
    [180, 100, 100, 180, 40, 40,   180, 100, 100, 180, 40, 40,   100,
     60, 50, 50, 60, 50,   60, 50, 50, 60, 50,   100, 100], dtype=np.float64)
KD = np.array(
    [5.0, 3.0, 3.0, 5.0, 0.3, 0.3,   5.0, 3.0, 3.0, 5.0, 0.3, 0.3,   5.0,
     0.3, 0.3, 0.3, 0.3, 0.3,   0.3, 0.3, 0.3, 0.3, 0.3,   1.0, 1.0], dtype=np.float64)
ACTION_SCALE = np.array(
    [0.5, 0.2, 0.2, 0.5, 0.5, 0.2,   0.5, 0.2, 0.2, 0.5, 0.5, 0.2,
     0.2, 0.2, 0.05, 0.2, 0.05,   0.2, 0.2, 0.05, 0.2, 0.05], dtype=np.float64)
OBS_SCALE = np.concatenate([
    np.full(22, 1.0), np.full(22, 0.05), np.full(22, 1.0),
    np.full(3, 1.0), np.full(3, 1.0),
])
CMD_SCALE = np.array([2.0, 2.0, 1.0])
OBS_CLIP = 100.0
ACTION_CLIP = 100.0
HISTORY = 15
CONTROL_DT = 0.01
TIMESTEP = 0.002
FRAME_SKIP = int(CONTROL_DT / TIMESTEP)
BASE_Z = 1.02


def load_model():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def build_addresses(model):
    qadr = np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES], dtype=int)
    dadr = np.array([model.joint(n).dofadr[0] for n in JOINT_NAMES], dtype=int)
    aid = np.array([model.actuator("motor_" + n).id for n in JOINT_NAMES], dtype=int)
    tau_lim = model.actuator_ctrlrange[aid][:, 1].copy()
    return {"qadr": qadr, "dadr": dadr, "aid": aid, "tau_lim": tau_lim}


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pd_step(model, data, addr, q_des, kp=KP, kd=KD):
    q = data.qpos[addr["qadr"]]
    qd = data.qvel[addr["dadr"]]
    tau = kp * (q_des - q) + kd * (0.0 - qd)
    tau = np.clip(tau, -addr["tau_lim"], addr["tau_lim"])
    data.ctrl[addr["aid"]] = tau
    mujoco.mj_step(model, data)
    return tau


class MnnPolicy:
    """封装 MNN 策略加载与推理：输入 (1,1083)，输出 (1,22)。"""

    def __init__(self, path=POLICY_PATH):
        import MNN
        self._MNN = MNN
        self.net = MNN.Interpreter(str(path))
        self.session = self.net.createSession()
        self.input = self.net.getSessionInput(self.session)
        self.output = self.net.getSessionOutput(self.session)

    def infer(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        MNN = self._MNN
        ih = MNN.Tensor(self.input.getShape(), MNN.Halide_Type_Float, obs,
                        MNN.Tensor_DimensionType_Caffe)
        self.input.copyFrom(ih)
        self.net.runSession(self.session)
        oh = MNN.Tensor(self.output.getShape(), MNN.Halide_Type_Float,
                        np.zeros(self.output.getShape(), np.float32),
                        MNN.Tensor_DimensionType_Caffe)
        self.output.copyToHostTensor(oh)
        return np.asarray(oh.getData(), np.float32).reshape(-1)


class Walker:
    """把「观测 1083 → MNN 推理 → 动作 22 → 目标角 → PD → mj_step」串成一步。"""

    def __init__(self, model=None, data=None):
        self.model, self.data = (model, data) if model is not None else load_model()
        self.addr = build_addresses(self.model)
        self.policy = MnnPolicy()
        self.active_idx = np.array([JOINT_NAMES.index(n) for n in ACTIVE_JOINT_NAMES],
                                   dtype=int)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = BASE_Z
        self.data.qpos[self.addr["qadr"]] = DEFAULT_Q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(len(self.active_idx))
        self.history = [self._obs() for _ in range(HISTORY)]

    def _obs(self):
        q = self.data.qpos[self.addr["qadr"]]
        qd = self.data.qvel[self.addr["dadr"]]
        R = quat_to_rot(self.data.qpos[3:7])
        proj_g = R.T @ np.array([0.0, 0.0, -1.0])
        ang = self.data.sensor("imu_angular_velocity").data.copy()
        obs = np.concatenate([
            q[self.active_idx] - DEFAULT_Q[self.active_idx],
            qd[self.active_idx],
            self.prev_action,
            ang,
            proj_g,
        ])
        return np.clip(obs * OBS_SCALE, -OBS_CLIP, OBS_CLIP)

    def step(self, command=(0.2, 0.0, 0.0), kp_ratio=1.0):
        obs = self._obs()
        self.history.append(obs)
        self.history.pop(0)
        full = np.concatenate(self.history + [np.asarray(command) * CMD_SCALE])
        action = np.clip(self.policy.infer(full), -ACTION_CLIP, ACTION_CLIP)
        q_des = DEFAULT_Q.copy()
        q_des[self.active_idx] += action * ACTION_SCALE
        kp = KP * kp_ratio
        kd = KD * kp_ratio
        for _ in range(FRAME_SKIP):
            pd_step(self.model, self.data, self.addr, q_des, kp, kd)
        self.prev_action = action
        return {"obs": obs, "action": action, "q_des": q_des}


def main():
    w = Walker()

    # 执行一个策略步，内部完成：观测 -> 推理 -> action -> q_des -> PD -> mj_step
    res = w.step(command=(0.2, 0.0, 0.0))
    action = res["action"]
    q_des = res["q_des"]

    print("动作 action（22 维，归一化）：")
    print("  ", np.round(action[:6], 4), "...")
    print("目标角偏移 = action × action_scale：")
    print("  ", np.round((action * ACTION_SCALE)[:6], 4), "...")
    print("目标角 q_des = default_q + action × action_scale：")
    print("  ", np.round(q_des[:6], 4), "...")
    print("默认姿态 default_q（前 6 关节）：", np.round(DEFAULT_Q[:6], 4))
    print("\n随后 PD：tau = kp*(q_des - q) + kd*(0 - qd)，裁剪到力矩范围后写入 ctrl。")


if __name__ == "__main__":
    main()
