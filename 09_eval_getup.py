"""09 — 评估训练好的「深蹲→站立」策略。

加载 08 训练出的权重，headless 统计（站起成功率、平均最高高度、终点高度、长度），
或 viewer 观看机器人从随机深蹲姿态站起并站住（回合结束自动开新回合）。

用法：
  python 09_eval_getup.py --ckpt checkpoints_getup/ppo_getup_final.pt --episodes 20
  python 09_eval_getup.py --ckpt checkpoints_getup/ppo_getup_final.pt --viewer
  python 09_eval_getup.py --ckpt checkpoints_getup/ppo_getup_final.pt --viewer --seconds 600
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

import mujoco
import mujoco.viewer  # MuJoCo 4.0 需显式导入，否则 mujoco.viewer 属性不存在
import numpy as np

import t800_common as tc


def make_env(level: int = 3):
    return importlib.import_module("07_get_up").T800GetUpEnv(level=level)


def make_ppo():
    return importlib.import_module("05_ppo").PPO(45, 8)


def run_episode(env, ppo, seed: int):
    """跑一回合，返回 (是否成功站起, 长度, 最大高度, 终点高度, 最大倾角, 总奖励)。"""
    obs, _ = env.reset(seed=seed)
    max_h = obs[43] + 0.25
    max_tilt, total, got_success = 0.0, 0.0, False
    for t in range(env.max_steps):
        action = ppo.act_deterministic(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        h = obs[43] + 0.25
        g = tc.projected_gravity(env.data)
        tilt = float(np.arccos(np.clip(-g[2], -1.0, 1.0)))
        max_h = max(max_h, h)
        max_tilt = max(max_tilt, tilt)
        if terminated and info.get("success"):
            got_success = True
        if terminated or truncated:
            return got_success, t + 1, max_h, h, max_tilt, total
    return got_success, env.max_steps, max_h, obs[43] + 0.25, max_tilt, total


def eval_headless(env, ppo, episodes: int, seed: int = 0):
    succ, lens, max_hs, end_hs, max_tilts, rets = 0, [], [], [], [], []
    for i in range(episodes):
        ok, length, max_h, end_h, max_tilt, ret = run_episode(env, ppo, seed=seed + i)
        succ += int(ok)
        lens.append(length)
        max_hs.append(max_h)
        end_hs.append(end_h)
        max_tilts.append(max_tilt)
        rets.append(ret)

    dt = env.model.opt.timestep * env.frame_skip  # 每策略步仿真时长
    print(f"episodes          : {episodes}")
    print(f"站起成功率(进成功带): {succ}/{episodes} = {100*succ/episodes:.0f}%")
    print(f"平均长度          : {np.mean(lens):.0f}步 ({np.mean(lens)*dt:.1f}s，满分 {env.max_steps*dt:.0f}s)")
    print(f"平均最高高度      : {np.mean(max_hs):.3f}m (最差 {np.min(max_hs):.3f}m，站立目标 1.016m)")
    print(f"平均终点高度      : {np.mean(end_hs):.3f}m")
    print(f"最大倾角          : {np.rad2deg(np.max(max_tilts)):.1f}°")
    print(f"平均回合奖励      : {np.mean(rets):.1f}")
    return succ / episodes


def run_viewer(env, ppo, seconds: float):
    env.reset()
    t0 = time.monotonic()
    n_episodes = 1
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running() and (time.monotonic() - t0) < seconds:
            obs = env._obs()
            action = ppo.act_deterministic(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            viewer.sync()
            step += 1
            if terminated or truncated:          # 站起成功或失败都开新回合
                env.reset()
                step = 0
                n_episodes += 1
    print(f"viewer 结束：运行 {time.monotonic() - t0:.1f}s（目标 {seconds:.0f}s），共 {n_episodes} 回合", flush=True)
    # MuJoCo 4.0 passive viewer 退出时 GLFW 双重 terminate 会段错误（见 06 的注释），
    # 跳过 atexit 直接退出进程。
    time.sleep(0.5)
    os._exit(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints_getup/ppo_getup_final.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--level", type=int, default=3, help="评估课程档（0 浅蹲 ~ 3 全混）")
    p.add_argument("--seconds", type=float, default=600.0, help="viewer 观看时长（真实秒）")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = make_env(level=args.level)
    ppo = make_ppo()
    ppo.load(args.ckpt)
    print(f"已加载策略：{args.ckpt}（评估档 L{args.level}）")

    if args.viewer:
        run_viewer(env, ppo, args.seconds)
    else:
        eval_headless(env, ppo, args.episodes, args.seed)


if __name__ == "__main__":
    main()
