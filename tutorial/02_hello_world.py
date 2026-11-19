"""第 2 课 · MuJoCo 入门：方块自由落体 Hello World。

用一个内联 XML 在 MuJoCo 世界里搭一个方块，让它自由落体，每隔 0.2 秒打印高度。
对应视频 P02 的「01 号程序」。

运行：python tutorial/02_hello_world.py
"""
import mujoco

XML = """
<mujoco model="block_drop">
    <option gravity="0 0 -9.81" timestep="0.002"/>
    <worldbody>
        <geom type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="block" pos="0 0 1">
            <freejoint/>
            <geom name="block_geom" type="box" size="0.05 0.05 0.05"
                  rgba="0.2 0.6 0.9 1" mass="1"/>
        </body>
    </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    print_interval = 0.2   # 秒
    next_print_t = 0.0
    sim_time = 3.0

    while data.time < sim_time:
        if data.time + 1e-12 > next_print_t:
            z = float(data.qpos[2])   # 方块的 Z 位置（自由关节，qpos[2] 即高度）
            print(f"t={data.time:.3f}s, z={z:.6f}m")
            next_print_t += print_interval
        mujoco.mj_step(model, data)


if __name__ == "__main__":
    main()
