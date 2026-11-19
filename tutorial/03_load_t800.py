"""第 3 课 · 加载 T800 机器人。

把众擎官方 SDK 里的 t800.xml 加载进 MuJoCo，打印身体/关节/电机数量。
对应视频 P03 的「02/03 号程序」。

运行：python tutorial/03_load_t800.py
"""
import mujoco

import common as C


def main():
    model, data = C.load_model()

    print(f"模型文件：{C.MODEL_PATH}")
    print(f"身体部件 nbody = {model.nbody}")
    print(f"关节数量 njnt = {model.njnt}（含 1 个浮基自由关节 + 25 个转动关节）")
    print(f"电机数量 nu    = {model.nu}")
    print(f"广义位置 nq    = {model.nq}（7 个浮基位姿 + 25 个关节角 = 32）")
    print(f"广义速度 nv    = {model.nv}（6 个浮基速度 + 25 个关节速度 = 31）")

    print("\n25 个关节：")
    for i, name in enumerate(C.JOINT_NAMES):
        print(f"  {i:2d}  {name}")


if __name__ == "__main__":
    main()
