# Changelog

本项目版本记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/) 风格，版本号语义化。

## [v0.0.0] - 2026-11-27

键盘控制行走。

### Added
- MuJoCo 仿真 T800 人形机器人，加载 MNN 行走策略（22 关节，观测 1083 维）。
- 键盘遥控行走：viewer 中 ↑/↓ 前进后退、←/→ 转向、空格停止，带软启动。
- YAML 全量配置（`config/t800.yaml`）：关节/PD、观测缩放、时序、按键、出生点。
- 默认平地场景 `mujoco/scene/ground/ground.xml`，顶层入口 `mujoco/t800.xml` 组装 robot + scene。
