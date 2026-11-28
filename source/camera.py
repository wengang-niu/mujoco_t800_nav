"""头部 RGB 相机：MuJoCo 离屏渲染，返回 (H, W, 3) uint8 图像。

用法：
    from camera import HeadCamera
    with HeadCamera(model) as cam:
        rgb = cam.render(data)
"""
from __future__ import annotations

import mujoco
import numpy as np


class HeadCamera:
    """按名字绑定 MJCF 相机并离屏渲染 RGB。"""

    def __init__(self, model: mujoco.MjModel, name: str = "head",
                 width: int = 640, height: int = 480):
        self.name = name
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.name)
        return self.renderer.render()          # (height, width, 3) uint8

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
