"""第 9 课 · 力矩范围（control range）。

打印 25 个电机的力矩上限，演示用 np.clip 把超限力矩裁剪到安全范围。
对应视频 P09 的「13/14 号程序」。

运行：python tutorial/09_torque_limit.py
"""
import numpy as np

import common as C


def main():
    model, data = C.load_model()
    addr = C.build_addresses(model)

    print(f"{'ID':>2} {'关节':22s} {'ctrlrange':>18s}")
    for i, name in enumerate(C.JOINT_NAMES):
        lo, hi = model.actuator_ctrlrange[addr["aid"][i]]
        print(f"{i:2d} {name:22s} [{lo:>6.0f}, {hi:>6.0f}]")

    # 演示裁剪：假设 PD 算出 5000 N·m（明显超限），裁剪到该电机的上限
    tau_raw = np.full(25, 5000.0)
    tau_safe = np.clip(tau_raw, -addr["tau_lim"], addr["tau_lim"])
    print("\n未裁剪 tau（前 3 项）：", tau_raw[:3])
    print("裁剪后 tau（前 3 项）：", tau_safe[:3], "（被压到各电机 ctrlrange 上限）")


if __name__ == "__main__":
    main()
