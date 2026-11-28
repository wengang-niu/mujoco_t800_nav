# Changelog

本项目版本记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/) 风格，版本号语义化。

## [v0.0.1] - 2026-11-29

头部 RGB 相机与 OpenCV 实时预览。

### Added
- 头部 RGB 相机 `head`：挂在 `LINK_HEAD_YAW` 上，朝机体 +x、`fovy=60`；viewer 的 Rendering 面板可切第一视角。
- `source/camera.py` 的 `HeadCamera`：基于 `mujoco.Renderer` 的离屏 RGB 接口，返回 `(H, W, 3)` uint8 图像。
- OpenCV 实时相机窗口：`source/main.py` 把 qpos 写到 `/tmp/t800_qpos.bin`，独立运行 `source/camera_view.py` 即可显示相机画面。

### Changed
- 遥控按键步长 `teleop.cmd_step` 由 0.3 调整为 0.1，方向键调节更精细。

## [v0.0.0] - 2026-11-27

键盘控制行走。

### Added
- MuJoCo 仿真 T800 人形机器人，加载 MNN 行走策略（22 关节，观测 1083 维）。
- 键盘遥控行走：viewer 中 ↑/↓ 前进后退、←/→ 转向、空格停止，带软启动。
- YAML 全量配置（`config/t800.yaml`）：关节/PD、观测缩放、时序、按键、出生点。
- 默认平地场景 `mujoco/scene/ground/ground.xml`，顶层入口 `mujoco/t800.xml` 组装 robot + scene。
