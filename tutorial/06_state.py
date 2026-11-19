"""第 6 课 · T800 的状态参数。

打印 nq / nv / nu 以及 qpos / qvel / ctrl 数组的长度与初始内容。
对应视频 P06 的「10 号程序」。

运行：python tutorial/06_state.py
"""
import mujoco
import numpy as np

import common as C


def main():
    model, data = C.load_model()

    print(f"nq = {model.nq}（qpos 长度，广义位置）")
    print(f"nv = {model.nv}（qvel 长度，广义速度）")
    print(f"nu = {model.nu}（ctrl 长度，执行器控制量）")

    print("\nqpos[:7]（浮基位姿：位置3 + 四元数4）：", np.round(data.qpos[:7], 4))
    print("qpos 关节部分（25 个关节角，初始全 0）：", np.round(data.qpos[7:], 4))
    print("qvel（初始全 0）：", np.round(data.qvel, 4))
    print("ctrl（初始全 0）：", np.round(data.ctrl, 4))
    print("\n读取一个关节的当前角：",
          data.qpos[model.jnt_qposadr[
              mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "J03_KNEE_PITCH_L")]])


if __name__ == "__main__":
    main()
