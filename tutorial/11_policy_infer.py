"""第 11 课 · AI 策略生成关节运动指令（MNN 推理）。

加载官方 MNN 策略模型，构造一次 1083 维观测，推理得到 22 维动作。
对应视频 P11 的「17 号程序」。

运行：python tutorial/11_policy_infer.py
"""
import numpy as np

import common as C


def main():
    w = C.Walker()            # 内部已加载模型 + MNN 策略
    policy = w.policy

    # 构造一次 1083 维输入：15 帧历史（每帧 72 维）+ 3 维指令
    obs = np.concatenate(w.history + [np.array([0.2, 0.0, 0.0]) * C.CMD_SCALE])
    print(f"观测维度 = {obs.shape[0]}（= 15×72 + 3 = 1083）")

    action = policy.infer(obs)
    print(f"动作维度 = {action.shape[0]}（22 个受控关节）")
    print("前 6 个动作：", np.round(action[:6], 6))
    print("动作范围：", round(float(action.min()), 4), "~", round(float(action.max()), 4))


if __name__ == "__main__":
    main()
