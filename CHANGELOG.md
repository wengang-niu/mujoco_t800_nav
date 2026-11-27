# Changelog

本项目版本记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/) 风格，版本号语义化。

## [Unreleased]

### Changed
- 移除咖啡馆场景 `mujoco/scenes/cafe_37874/` 与场景家具资源 `mujoco/objects/`（及 `doc/scene.png`），
  默认环境改为平地 `environment/ground.xml`，出生点 `reset.base_pos` 改回 `[0.0, 0.0]`。

### Removed
- `--check` 无显示冒烟测试模式，只保留 viewer 遥控一种用法。

## [v0.0.0] - 2026-11-25

首次发布：键盘控制行走。

### Added
- MuJoCo 仿真 T800 人形机器人，加载 MNN 行走策略（22 关节，观测 1083 维）。
- 键盘遥控行走：viewer 中 ↑/↓ 前进后退、←/→ 转向、空格停止，带软启动。
- 咖啡馆场景 `cafe_37874`（sales floor 128㎡ + bakery kitchen 等）及俯视图。
- YAML 全量配置（`config/t800.yaml`）：关节/PD、观测缩放、时序、按键、出生点。
- `--check` 无显示冒烟测试。
- `tools/fetch_scene_assets.py` 重建 `mujoco/objects/` 大体积场景资源。
