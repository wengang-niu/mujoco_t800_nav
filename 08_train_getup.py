"""08 — 训练 07 的「深蹲→站立」策略（复用 05 的手写 PPO）。

训练设计（可行性研究已实测验证）：
- 环境 07_get_up.T800GetUpEnv：8 维绝对目标角（髋/膝/踝 pitch + 踝 roll ×2），
  比例式目标低通（0.6 rad/s，所有关节同步到站——统一速率会让踝提前归零翻倒），
  Δh 高度进步奖励（马尔可夫，v4）+ 带内 upright/feet 收割（P1 防 hacking）+
  弱升程 upright 指引 0.02 + 带内镇定整形 −0.4·mean(|a|) − 0.2·足力不对称
  − 0.3·mean(q²)（v4d：v4c 实测 L2 4%/L3 60%，惩罚 1/10 收割量级无法撼动 crank
  吸引子，评审 workflow 裁定加码）+ 成功带累计 150 步判定（回合内计数、出带不清零，
  防 flicker 收割循环 → +20 提前终局）。
- 课程：L0 浅蹲(k 0.10~0.45) → L1 中蹲(0.45~0.85) → L2 深蹲(0.85~1.20) → L3 全混，
  每 400 迭代升一档（--no-curriculum 直接全混）。
- 每 200 迭代确定性评估 10 回合（固定种子）：站起成功率（进成功带）、平均最高高度、
  平均终点高度、平均回合长度。
- CPU 实测 07 环境 ~2000 env.step/s：2048 步/iter 收集约 1s + PPO 更新约 0.5s，
  2000 迭代 ≈ 50 分钟（8 环境顺序步进）。

验收标准（可行性验证训练）：
  1) iter 300 时 eval 平均最高高度 > 0.80m（学到「先站起来」）
  2) 最终 eval 站起成功率 ≥ 70%（站住 1.5s 进成功带）

用法：
  python 08_train_getup.py --iters 2000          # 完整训练（默认）
  python 08_train_getup.py --iters 200 --smoke   # 快速冒烟 + 05 的 PPO 自检
  python 08_train_getup.py --eval checkpoints_getup/ppo_getup_final.pt   # 仅评估
  python 08_train_getup.py --gpu                 # 网络放 GPU（物理仍在 CPU）
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

import numpy as np
import torch

ppo_mod = importlib.import_module("05_ppo")
PPO = ppo_mod.PPO
collect_rollout_vec = ppo_mod.collect_rollout_vec
smoke_test = ppo_mod.smoke_test

OBS_DIM, ACT_DIM = 45, 8


def make_env(level: int = 3):
    return importlib.import_module("07_get_up").T800GetUpEnv(level=level)


def make_ppo() -> PPO:
    ppo = PPO(OBS_DIM, ACT_DIM, ent_coef=0.002)
    # 绝对目标角任务动作幅度大：初始 std 取 ~0.45，鼓励早期大范围探索目标姿态
    ppo.ac.log_std.data.fill_(-0.8)
    return ppo


class VecGetupEnv:
    """进程内多环境：n 个 T800 实例顺序步进，供批量采样摊薄网络开销（同 05 结构）。"""

    def __init__(self, n: int, level: int):
        self.envs = [make_env(level=level) for _ in range(n)]
        self.n = n
        self.obs = np.stack([e.reset()[0] for e in self.envs]).astype(np.float32)

    def set_level(self, level: int) -> None:
        for e in self.envs:
            e.set_level(level)

    def set_levels(self, levels) -> None:
        for e, lv in zip(self.envs, levels):
            e.set_level(lv)

    def step(self, actions: np.ndarray):
        obs_list, rew = [], np.zeros(self.n)
        dones = np.zeros(self.n, dtype=bool)
        for i, e in enumerate(self.envs):
            o, r, term, trunc, _ = e.step(actions[i])
            d = term or trunc
            if d:
                o, _ = e.reset()
            obs_list.append(o)
            rew[i] = r
            dones[i] = d
        self.obs = np.stack(obs_list).astype(np.float32)
        return self.obs, rew, dones


def eval_getup(ppo: PPO, episodes: int = 10, level: int = 3, seed: int = 1234) -> dict:
    """确定性评估：固定种子，全混档（L3）。返回指标字典。"""
    env = make_env(level=level)
    succ, max_hs, end_hs, lens, rets = [], [], [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        ep_ret, max_h, end_h, got_success = 0.0, obs[43] + 0.25, obs[43] + 0.25, False
        for t in range(env.max_steps):
            action = ppo.act_deterministic(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            h = obs[43] + 0.25
            max_h = max(max_h, h)
            end_h = h
            if terminated and info.get("success"):
                got_success = True
            if terminated or truncated:
                break
        succ.append(got_success)
        max_hs.append(max_h)
        end_hs.append(end_h)
        lens.append(t + 1)
        rets.append(ep_ret)
    return {"success_rate": float(np.mean(succ)),
            "max_h_mean": float(np.mean(max_hs)),
            "max_h_min": float(np.min(max_hs)),
            "end_h_mean": float(np.mean(end_hs)),
            "len_mean": float(np.mean(lens)),
            "ret_mean": float(np.mean(rets))}


def print_eval(tag: str, m: dict) -> None:
    print(f"[eval {tag}] 成功率={m['success_rate']*100:.0f}% | 平均最高高度={m['max_h_mean']:.3f}m "
          f"(最差={m['max_h_min']:.3f}) | 平均终点高度={m['end_h_mean']:.3f}m | "
          f"平均长度={m['len_mean']:.0f}步 | 平均回报={m['ret_mean']:.1f}", flush=True)


# 课程每档时长（迭代数）与每档的环境分配：
# L2 深蹲档混合一半 L3 环境——纯深蹲训练会学出「站起-翻倒循环」hacking（实测），
# 需要成功回合经验持续注入。L2 档期深蹲回合占比 ≈ 4/8 + 4/8×0.32 ≈ 66%。
CURRICULUM_STEPS = [300, 300, 800, 800]   # L0 300 + L1 300 + L2 800 + L3 800
CURRICULUM_MIX = {0: [0] * 8, 1: [1] * 8, 2: [2] * 4 + [3] * 4, 3: [3] * 8}
CURRICULUM_MAX = 3


def train(ppo: PPO, iters: int, n_steps: int, nenvs: int, save_dir: str,
          curriculum: bool, start_level: int = 0, eval_every: int = 200,
          log_every: int = 20) -> None:
    os.makedirs(save_dir, exist_ok=True)
    level = start_level    # 固定档（--no-curriculum）或课程起始档
    venv = VecGetupEnv(nenvs, level=level)
    if curriculum:
        venv.set_levels(CURRICULUM_MIX[level])
    print(f"device={ppo_mod.DEVICE}  iters={iters}  n_steps={n_steps}  nenvs={nenvs}  "
          f"curriculum={curriculum} start_level={level}", flush=True)
    for it in range(1, iters + 1):
        t0 = time.time()
        rollout, ep_rewards, ep_lens = collect_rollout_vec(venv, ppo, n_steps)
        stats = ppo.train_iteration(rollout)
        mean_ep = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        if it % log_every == 0:
            print(f"iter {it:5d} | L{level} mean_ep_return={mean_ep:8.1f} mean_len={mean_len:6.1f} | "
                  f"policy_loss={stats['policy_loss']:7.4f} value_loss={stats['value_loss']:7.4f} "
                  f"entropy={stats['entropy']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if curriculum:
            # 从当前档开始累计每档时长（支持 --start-level 续训时计时重定位）
            acc = 0
            for lv in range(start_level, CURRICULUM_MAX):
                acc += CURRICULUM_STEPS[lv]
                if level == lv and it >= acc:
                    level = lv + 1
                    venv.set_levels(CURRICULUM_MIX[level])
                    print(f"== 课程升档 → L{level}（环境分配 {CURRICULUM_MIX[level]}） ==", flush=True)
        if it % eval_every == 0:
            print_eval(f"iter{it}/L2", eval_getup(ppo, episodes=10, level=2))
            print_eval(f"iter{it}/L3", eval_getup(ppo, episodes=10, level=3))
        if it % 200 == 0:
            ppo.save(os.path.join(save_dir, f"ppo_getup_{it}.pt"))
    ppo.save(os.path.join(save_dir, "ppo_getup_final.pt"))
    print("saved:", os.path.join(save_dir, "ppo_getup_final.pt"), flush=True)
    print_eval("final", eval_getup(ppo))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=2000, help="训练迭代数")
    p.add_argument("--nsteps", type=int, default=2048, help="每迭代收集步数（总环境步 = iters×nsteps）")
    p.add_argument("--nenvs", type=int, default=8, help="并行环境数")
    p.add_argument("--gpu", action="store_true", help="网络放 GPU（MuJoCo 物理仍在 CPU）")
    p.add_argument("--no-curriculum", action="store_true", help="跳过课程，直接全混档 L3")
    p.add_argument("--smoke", action="store_true", help="训练前先跑 05 的 PPO 自检")
    p.add_argument("--eval", type=str, default="", help="只评估指定 checkpoint")
    p.add_argument("--resume", type=str, default="", help="从 checkpoint 续训")
    p.add_argument("--start-level", type=int, default=0, help="课程起始档（配合 --resume）")
    p.add_argument("--save-dir", type=str, default="checkpoints_getup", help="checkpoint 保存目录")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu:
        ppo_mod.DEVICE = torch.device("cuda")
        print("GPU 训练：网络在 cuda，物理仍在 CPU", flush=True)

    if args.smoke:
        smoke_test()

    ppo = make_ppo()
    if args.eval:
        ppo.load(args.eval)
        print(f"已加载策略：{args.eval}")
        print_eval("loaded", eval_getup(ppo, episodes=20))
        return
    if args.resume:
        ppo.load(args.resume)
        print(f"续训起点：{args.resume}，课程从 L{args.start_level} 起")

    train(ppo, args.iters, args.nsteps, args.nenvs, args.save_dir,
          curriculum=not args.no_curriculum, start_level=args.start_level)


if __name__ == "__main__":
    main()
