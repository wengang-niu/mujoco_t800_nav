"""OpenCV 实时相机窗口（独立脚本）。

终端 1：python source/main.py
终端 2：python source/camera_view.py

main.py 把 qpos 写入 /tmp/t800_qpos.bin；本脚本读取该文件，用
mujoco.Renderer 渲染 head 相机并显示。相机与 viewer 分属两个进程，
避免 GLFW 与 EGL 两套 OpenGL 上下文在同一进程里冲突（会段错误）。

退出：按 Esc/q，或关闭窗口。
"""
from __future__ import annotations

import pathlib
import time

import cv2
import mujoco
import numpy as np

from camera import HeadCamera
from config import load_config

STATE_PATH = pathlib.Path("/tmp/t800_qpos.bin")
WINDOW = "T800 head camera"


def main():
    cfg = load_config()
    model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
    data = mujoco.MjData(model)

    print(f"读取 {STATE_PATH}；若长时间无画面，先运行 python source/main.py", flush=True)
    nbytes = model.nq * np.dtype(np.float64).itemsize
    while True:
        try:
            if STATE_PATH.stat().st_size == nbytes:
                break
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    state = np.memmap(STATE_PATH, dtype=np.float64, mode="r", shape=(model.nq,))

    with HeadCamera(model) as cam:
        while True:
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            rgb = cam.render(data)
            cv2.imshow(WINDOW, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
