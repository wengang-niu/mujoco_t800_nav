"""12 — 评估训练好的「地面折叠蹲→深蹲交接」策略。

加载 11 训练出的权重，headless 统计（交接成功率、平均最高骨盆高、终点高、长度），
或 viewer 观看机器人从折叠蹲 tuck 试图站到深蹲交接带。

用法：
  python 12_eval_getup_ground.py --ckpt checkpoints_getup_ground/ppo_gg_final.pt --episodes 20
  python 12_eval_getup_ground.py --ckpt checkpoints_getup_ground/ppo_gg_final.pt --viewer
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


def make_env():
    return importlib.import_module("10_get_up_ground").T800GetUpGroundEnv()


def make_ppo():
    return importlib.import_module("05_ppo").PPO(62, 12)


def run_episode(env, ppo, seed: int, start: str = "tuck"):
    """跑一回合，返回 (是否交接成功, 长度, 最大骨盆高, 终点骨盆高, 最大倾角, 总奖励)。"""
    obs, _ = env.reset(seed=seed, options={"start": start})
    max_z = obs[47] + 0.25
    max_tilt, total, got_success = 0.0, 0.0, False
    for t in range(env.max_steps):
        action = ppo.act_deterministic(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        z = obs[47] + 0.25
        g = tc.projected_gravity(env.data)
        tilt = float(np.arccos(np.clip(-g[2], -1.0, 1.0)))
        max_z = max(max_z, z)
        max_tilt = max(max_tilt, tilt)
        if terminated and info.get("success"):
            got_success = True
        if terminated or truncated:
            return got_success, t + 1, max_z, z, max_tilt, total
    return got_success, env.max_steps, max_z, obs[47] + 0.25, max_tilt, total


def eval_headless(env, ppo, episodes: int, seed: int = 0, start: str = "tuck"):
    succ, lens, max_zs, end_zs, max_tilts, rets = 0, [], [], [], [], []
    for i in range(episodes):
        ok, length, max_z, end_z, max_tilt, ret = run_episode(env, ppo, seed=seed + i, start=start)
        succ += int(ok)
        lens.append(length)
        max_zs.append(max_z)
        end_zs.append(end_z)
        max_tilts.append(max_tilt)
        rets.append(ret)

    dt = env.model.opt.timestep * env.frame_skip  # 每策略步仿真时长
    print(f"episodes          : {episodes}")
    print(f"交接成功率(进交接带): {succ}/{episodes} = {100*succ/episodes:.0f}%")
    print(f"平均长度          : {np.mean(lens):.0f}步 ({np.mean(lens)*dt:.1f}s，满分 {env.max_steps*dt:.0f}s)")
    print(f"平均最高骨盆高    : {np.mean(max_zs):.3f}m (最差 {np.min(max_zs):.3f}m，交接需 0.87m、开环上限 0.63m)")
    print(f"平均终点骨盆高    : {np.mean(end_zs):.3f}m")
    print(f"最大倾角          : {np.rad2deg(np.max(max_tilts)):.1f}°")
    print(f"平均回合奖励      : {np.mean(rets):.1f}")
    return succ / episodes


def run_viewer(env, ppo, seconds: float, start: str = "tuck"):
    env.reset(options={"start": start})
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
            if terminated or truncated:          # 成功或失败都开新回合
                env.reset(options={"start": start})
                step = 0
                n_episodes += 1
    print(f"viewer 结束：运行 {time.monotonic() - t0:.1f}s（目标 {seconds:.0f}s），共 {n_episodes} 回合", flush=True)
    # MuJoCo 4.0 passive viewer 退出时 GLFW 双重 terminate 会段错误（见 06 的注释），
    # 跳过 atexit 直接退出进程。
    time.sleep(0.5)
    os._exit(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints_getup_ground/ppo_gg_final.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--seconds", type=float, default=600.0, help="viewer 观看时长（真实秒）")
    p.add_argument("--start", type=str, default="tuck", choices=["tuck", "squat"],
                   help="起点模式：tuck 地面主课 / squat 带内技能课")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = make_env()
    ppo = make_ppo()
    ppo.load(args.ckpt)
    print(f"已加载策略：{args.ckpt}")

    if args.viewer:
        run_viewer(env, ppo, args.seconds, start=args.start)
    else:
        eval_headless(env, ppo, args.episodes, args.seed, start=args.start)


if __name__ == "__main__":
    main()
