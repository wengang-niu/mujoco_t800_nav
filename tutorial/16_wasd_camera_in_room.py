"""第 16 课变体 · 遥控行走 + 第一人称相机 + 室内场景(ProcTHOR)。

把 T800 放进 LightNav 的 ProcTHOR 室内场景(val_2)，用 WASD 在房间里遥控行走，
按 C 切换第一人称(头部相机) / 自由视角。

基于 16_wasd.py / 16_wasd_camera.py，改动点：
  1. 用 MjSpec 把 T800 机器人(serial_t800.xml) attach 进室内场景(val_2_ceiling.xml)；
  2. attach 会加名字前缀 PREFIX，所有按名查找都要加前缀；
  3. 基座 freejoint 的 qpos 地址动态查找，不再写死 data.qpos[2]/[3:7]。

运行：python tutorial/16_wasd_camera_in_room.py
"""
import pathlib
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer  # 需显式导入，否则 mujoco.viewer 不存在
import numpy as np

# ---- 路径（本文件在 tutorial/ 下，上一级是项目根目录）----
ROOT = pathlib.Path(__file__).resolve().parent.parent
SDK = ROOT / "engineai_robotics_native_sdk"
ROBOT_PATH = (SDK / "assets" / "resource" / "robot" / "t800" / "xml"
              / "serial_t800.xml")                    # 机器人本体（不含地面环境）
SCENE_PATH = (ROOT / "LightNav-0" / "mujoco_demo" / "vln_mujoco" / "assets"
              / "scenes" / "procthor-10k-val" / "val_2_ceiling.xml")
POLICY_PATH = (SDK / "assets" / "config" / "t800" / "rl_walking_example" / "policy"
               / "t800_260618_165257_30000.mnn")

PREFIX = "t800_"          # attach 时给机器人所有 body/joint/actuator/sensor 加的前缀
SPAWN = (6.5, 13.8)       # 室内场景里的生成点 (x, y)，z 用 BASE_Z
KEEP_CEILING = False      # 是否保留房顶；False 时移除房顶，方便俯视室内

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

# GLFW 键码（MuJoCo viewer 的 key_callback 传入的即 GLFW 键码）
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_SPACE = 262, 263, 264, 265, 32
KEY_C = 67   # C 键：切换 第一人称/自由视角


def load_scene_element(source):
    """加载室内场景 XML 并预处理（复用 LightNav 的做法）。

    1) 冻结环境：删掉场景里所有 joint/freejoint（门等固定住），
       让合并后模型里唯一的自由关节就是 T800 的 freejoint；
    2) 把资源 file 路径转成绝对路径，保证任意工作目录都能加载；
    3) 设物理步长为 TIMESTEP。
    """
    source = pathlib.Path(source).resolve()
    root = ET.parse(source).getroot()
    for parent in root.iter():
        for child in tuple(parent):
            if child.tag in {"joint", "freejoint"}:
                parent.remove(child)
    # 移除房顶（body 名以 ceiling 开头，纯视觉、不参与碰撞），便于俯视
    if not KEEP_CEILING:
        for parent in root.iter():
            for child in tuple(parent):
                name = child.get("name") or ""
                if child.tag == "body" and name.startswith("ceiling"):
                    parent.remove(child)
    for element in root.iter():
        filename = element.get("file")
        if filename:
            element.set("file", str((source.parent / filename).resolve()))
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", str(TIMESTEP))
    return root


def load_model_with_room():
    """加载 ProcTHOR 场景，把 T800 挂进去，并在头部加第一人称相机。"""
    scene = mujoco.MjSpec.from_string(
        ET.tostring(load_scene_element(SCENE_PATH), encoding="unicode"))
    robot = mujoco.MjSpec.from_file(str(ROBOT_PATH))

    # 头部加第一人称相机（挂在 LINK_HEAD_YAW 上，跟着机器人走）
    head = robot.body("LINK_HEAD_YAW")
    cam = head.add_camera()
    cam.name = "head_cam"
    cam.pos = [0.12, 0.0, 0.02]
    # 相机沿局部 -z 看、局部 +y 是画面上方：-z 对准 +x(前)，+y 对准 +z(上)
    cam.quat = [0.5, 0.5, -0.5, -0.5]
    cam.fovy = 60
    cam.resolution = [640, 480]

    # 基座放场景里的生成点（z=BASE_Z 让脚底刚好触地）
    base = robot.body("LINK_BASE")
    base.pos = [SPAWN[0], SPAWN[1], BASE_Z]

    # 把机器人 attach 进场景（加前缀避免与场景重名）
    anchor = scene.worldbody.add_frame()
    scene.attach(robot, prefix=PREFIX, frame=anchor)

    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def build_addresses(model):
    """预计算 25 个关节的 qpos/qvel/执行器地址与力矩上限（名字带前缀）。"""
    qadr = np.array([model.joint(PREFIX + n).qposadr[0] for n in JOINT_NAMES], dtype=int)
    dadr = np.array([model.joint(PREFIX + n).dofadr[0] for n in JOINT_NAMES], dtype=int)
    aid = np.array([model.actuator(PREFIX + "motor_" + n).id for n in JOINT_NAMES], dtype=int)
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
        self.model, self.data = (model, data) if model is not None else load_model_with_room()
        self.addr = build_addresses(self.model)
        self.policy = MnnPolicy()
        self.active_idx = np.array([JOINT_NAMES.index(n) for n in ACTIVE_JOINT_NAMES],
                                   dtype=int)
        # 基座 freejoint 的地址（attach 后不再写死为 qpos 0~6）
        self.base_body_id = self.model.body(PREFIX + "LINK_BASE").id
        base_jnt = self.model.body_jntadr[self.base_body_id]
        self.base_qposadr = int(self.model.jnt_qposadr[base_jnt])
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # 基座位姿：xy 放生成点，z=BASE_Z，朝向单位四元数
        self.data.qpos[self.base_qposadr] = SPAWN[0]
        self.data.qpos[self.base_qposadr + 1] = SPAWN[1]
        self.data.qpos[self.base_qposadr + 2] = BASE_Z
        self.data.qpos[self.base_qposadr + 3:self.base_qposadr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.addr["qadr"]] = DEFAULT_Q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(len(self.active_idx))
        self.history = [self._obs() for _ in range(HISTORY)]

    def _obs(self):
        q = self.data.qpos[self.addr["qadr"]]
        qd = self.data.qvel[self.addr["dadr"]]
        R = quat_to_rot(self.data.xquat[self.base_body_id])   # 基座四元数
        proj_g = R.T @ np.array([0.0, 0.0, -1.0])
        ang = self.data.sensor(PREFIX + "imu_angular_velocity").data.copy()
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
    model, data = load_model_with_room()   # 带室内场景 + 第一人称相机的模型
    w = Walker(model, data)
    cmd = [0.0, 0.0, 0.0]
    step = 0.2            # 每次按键改变的速度（m/s 或 rad/s）
    transition = int(0.5 / CONTROL_DT)

    head_cam_id = model.camera(PREFIX + "head_cam").id   # 第一人称相机的 id
    view = {"fpv": False}                                # 当前是否第一人称视角

    def key_cb(keycode: int):
        nonlocal cmd
        if keycode == KEY_UP:      # ↑ 前进
            cmd[0] = min(cmd[0] + step, 0.6)
        elif keycode == KEY_DOWN:  # ↓ 后退
            cmd[0] = max(cmd[0] - step, -0.6)
        elif keycode == KEY_LEFT:  # ← 左转
            cmd[2] = min(cmd[2] + step, 1.0)
        elif keycode == KEY_RIGHT:  # → 右转
            cmd[2] = max(cmd[2] - step, -1.0)
        elif keycode == KEY_SPACE:  # 空格停止
            cmd = [0.0, 0.0, 0.0]
        elif keycode == KEY_C:      # C 切换第一人称 / 自由视角
            view["fpv"] = not view["fpv"]
            print("视角 → 第一人称 head_cam" if view["fpv"] else "视角 → 自由(第三人称)")
        print(f"command = [{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}]")

    with mujoco.viewer.launch_passive(w.model, w.data, key_callback=key_cb) as viewer:
        k = 0
        while viewer.is_running():
            # 按 view 标志切换渲染相机（加锁，避免与 viewer 渲染线程竞争）
            with viewer.lock():
                if view["fpv"]:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = head_cam_id   # 切到挂在头部的相机
                else:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            ratio = min(1.0, k / transition)
            w.step(command=cmd, kp_ratio=0.2 + 0.8 * ratio)
            viewer.sync()
            time.sleep(CONTROL_DT)
            k += 1


if __name__ == "__main__":
    main()
