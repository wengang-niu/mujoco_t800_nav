"""MNN 行走策略封装：输入 (1,1083)，输出 (22,) 动作。"""
from __future__ import annotations

import pathlib

import numpy as np


class MnnPolicy:
    """封装 MNN 策略加载与推理：输入 (1,1083)，输出 (1,22)。"""

    def __init__(self, path: pathlib.Path):
        import MNN                                  # 延迟导入，缺失时不影响其余代码
        self._MNN = MNN                             # 存模块引用供 infer 使用
        self.net = MNN.Interpreter(str(path))       # 加载 .mnn 模型（结构+权重）
        self.session = self.net.createSession()     # 建推理会话（前向计算工作区）
        self.input = self.net.getSessionInput(self.session)    # 输入插槽
        self.output = self.net.getSessionOutput(self.session)  # 输出插槽

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)  # → float32 (1,1083)
        MNN = self._MNN
        ih = MNN.Tensor(self.input.getShape(), MNN.Halide_Type_Float, obs,
                        MNN.Tensor_DimensionType_Caffe)        # 把观测包成 MNN 张量
        self.input.copyFrom(ih)                       # 写入网络输入
        self.net.runSession(self.session)             # 前向推理一次
        oh = MNN.Tensor(self.output.getShape(), MNN.Halide_Type_Float,
                        np.zeros(self.output.getShape(), np.float32),
                        MNN.Tensor_DimensionType_Caffe)        # 预分配输出容器
        self.output.copyToHostTensor(oh)              # 拷回宿主内存
        return np.asarray(oh.getData(), np.float32).reshape(-1)  # → (22,) 动作
