"""第 4 课 · 认识 T800 的 joint 和 link。

用 mj_id2name / mj_name2id 遍历所有关节（joint）和刚体连杆（body），
打印名称、关节类型与行程。对应视频 P04 的「05/06 号程序」。

运行：python tutorial/04_joints_links.py
"""
import pathlib

import mujoco

# 本文件在 tutorial/ 下，上一级是项目根目录。
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "engineai_robotics_native_sdk" / "assets" / "resource" / "t800.xml"


def load_model():
    """加载 T800 模型并复位（绝对路径，MuJoCo 4.0 下避免相对路径加载失败）。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def main():
    model, data = load_model()

    # --- 关节（joint）---
    print(f"== 关节（共 {model.njnt} 个，含浮基自由关节）==")
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        jtype = model.jnt_type[jid]      # 关节类型：0=自由(free) 1=球(ball) 2=滑动(slide) 3=铰链(hinge)
        qadr = model.jnt_qposadr[jid]    # 该关节在 data.qpos（广义位置）中的起始下标
        dadr = model.jnt_dofadr[jid]     # 该关节在 data.qvel（广义速度，按自由度编号）中的起始下标
        print(f"  jid={jid:2d}  name={name!r:24s} type={int(jtype)} "
              f"qpos_adr={qadr:2d} dof_adr={dadr:2d}")

    # --- 连杆（body）---
    print(f"\n== 连杆（共 {model.nbody} 个，含 world）==")
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        pos = model.body_pos[bid].copy()
        mass = model.body_mass[bid]
        print(f"  bid={bid:2d}  name={name!r:24s} pos={pos} mass={mass:.3f}")

    # --- 名称 -> ID 反向查找 ---
    knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "J03_KNEE_PITCH_L")
    print(f"\n名称 -> ID 示例：J03_KNEE_PITCH_L -> jid={knee_id}")


if __name__ == "__main__":
    main()
