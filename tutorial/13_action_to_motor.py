"""第 13 课 · AI 指令到机器人肌肉运动（动作映射）。

网络输出 22 维归一化动作，不能直接当力矩。先乘 action_scale 得到目标角偏移，
再叠加默认姿态得到目标角，最后经 PD 换算成力矩。对应视频 P13。

运行：python tutorial/13_action_to_motor.py
"""
import numpy as np

import common as C


def main():
    w = C.Walker()

    # 执行一个策略步，内部完成：观测 -> 推理 -> action -> q_des -> PD -> mj_step
    res = w.step(command=(0.2, 0.0, 0.0))
    action = res["action"]
    q_des = res["q_des"]

    print("动作 action（22 维，归一化）：")
    print("  ", np.round(action[:6], 4), "...")
    print("目标角偏移 = action × action_scale：")
    print("  ", np.round((action * C.ACTION_SCALE)[:6], 4), "...")
    print("目标角 q_des = default_q + action × action_scale：")
    print("  ", np.round(q_des[:6], 4), "...")
    print("默认姿态 default_q（前 6 关节）：", np.round(C.DEFAULT_Q[:6], 4))
    print("\n随后 PD：tau = kp*(q_des - q) + kd*(0 - qd)，裁剪到力矩范围后写入 ctrl。")


if __name__ == "__main__":
    main()
