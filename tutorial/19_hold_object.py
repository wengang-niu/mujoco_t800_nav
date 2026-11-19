"""第 19 课 · T800 手持物体走路（坐标绑定）。

用「坐标绑定」简化抓取：每个物理步把木块的世界坐标设为右手腕的世界坐标，
让木块跟随手腕一起移动（绕开摩擦/形变等真实抓取细节）。对应视频 P19 的「26 号程序」。

运行：python tutorial/19_hold_object.py
"""
import mujoco

import common as C


def main():
    model, data = C.load_model_with_box(half=0.06, mass=0.4, pos=(0.3, -0.2, 0.8),
                                        collidable=False)  # 展示块，不参与碰撞
    w = C.Walker(model, data)
    box_id = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "box")
    wrist_id = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "LINK_WRIST_END_R")
    box_qadr = w.model.jnt_qposadr[w.model.body_jntadr[box_id]]  # 木块自由关节 qpos 地址

    transition = int(0.5 / C.CONTROL_DT)
    total = int(8.0 / C.CONTROL_DT)
    print(f"{'t(s)':>6} {'wrist_x':>8} {'box_x':>8}")
    for k in range(total):
        ratio = min(1.0, k / transition)
        w.step(command=(0.2, 0.0, 0.0), kp_ratio=0.2 + 0.8 * ratio)
        # 坐标绑定：木块自由关节位置 = 右手腕世界位置
        w.data.qpos[box_qadr:box_qadr + 3] = w.data.xpos[wrist_id]
        if k % 100 == 0:
            print(f"{k * C.CONTROL_DT:6.2f} {w.data.xpos[wrist_id][0]:8.3f} "
                  f"{w.data.xpos[box_id][0]:8.3f}")
    print("结束：木块已随右手腕一起前进（坐标绑定生效）")


if __name__ == "__main__":
    main()
