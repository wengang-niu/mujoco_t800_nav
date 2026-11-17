"""11 — 训练 10 的「地面折叠蹲→深蹲交接」策略（复用 05 的手写 PPO）。

训练设计（承接 07/08 的 v4d 结论与 2026-11-14 地面站起可行性研究）：
- 环境 10_get_up_ground.T800GetUpGroundEnv：12 维绝对目标角（8 腿 + 4 臂），
  比例式目标低通（0.6 rad/s 全关节同步，07 实测防翻倒）、从折叠蹲 tuck 起步
  （最深可达地面姿态，pelvis ≈0.53）、成功 = 深蹲交接带（EMA z≥0.87、tilt≤0.35、
  双足力 >200N——防折叠小腿压足自接触伪造）累计 150 步 → +20 提前终局。
- 奖励 v2「记录势」重构（2026-11-14 总合成定稿）：progress=5Δz 已删除（对称增量
  伸缩和恒负 ≈−1.6/回合，日志实证压制升程：eval max_z 0.627→0.553、熵 7.6→11.4），
  替换为 r1=10×Δ⁺m_ema 记录信用（m_ema 单调封顶 0.87、入观测 59→62）+ r2 手锚
  记录信用（3×Δ⁺m_ema×hc，z<0.72 ∧ min fz>50）+ r3 CoM 守线 / r4 upright（仅记录
  创新步，0.03/0.02，z>0.65 ∧ min fz>50）+ 带内 v4d 原样 + 地板税 z<0.40→−0.02。
  升程尝试净 ≈+1~+2（旧 −1.6），四类作弊循环（弹射/躺地/静持/起倒）均构造性封顶。
- 奖励 v3（run 3 裁定，终审 workflow wf_8a7bf9bc-06a）：r1 10→12；r2 删除（手几何
  够不到地）；新增 r9 预带区块（ema_z≥0.72 ∧ min fz>150 ∧ com_fwd≥0 ∧ pre_cnt<150，
  步上限 0.30 回合封顶 150）；目标低通 z 调度 0.012/0.006（z=0.80 分界）。配合
  --warmstart（v4d 特征映射 + value_head 零重置 + log_std −1.0）构成 run 3：tuck
  单课 2000 iter、五分支终判、守卫 200/400/600、squat@pitch0 侧查每 400 iter。
- 课程表（--course）：
  A. --course squat：带内技能课（k=1.2 深蹲出生，实测落稳 z≈0.857 带边之下 1.3cm，
     教「进带+保持」；07 v4d 已实证该技能 ckpt1800 L2 96.5%）；
  B. --course tuck：地面主课（默认，膝先行主路径，全地面定论）；
  C. --course auto：A→B 自动切换 + iter800 救援 + iter1200 定论判据（见下）。
- 预登记判据（总合成定稿，全部仅打印/切换课程，不自动杀训练——判死刑必须人工复核）：
  iter800：eval 成功率=0 ∧ max_z<0.68 → 切换 squat 救援课程 400 迭代（补 0.69→0.87
    过墙段技能，再回 tuck 定论）。
  iter1200（回到 tuck 后）：
    ① 成功率>0 → 继续完整验收；跨全期扫 checkpoint 选点（MEMORY 教训：v4d 峰值在
       iter1800 之后退化，单点评估会误判）。
    ② 成功率=0 ∧ max_z<0.75 → 打印「硬件瓶颈告警」（0.69 墙：踝 35 N·m CoP ±8.4cm
       /无腰俯仰/踝 ±39°/摩擦 136N<拉链 300-550N），iter2000 复核后如实上报。
    ③ 成功率=0 ∧ ema_end∈[0.68,0.82) ∧ max_z>0.8 → 打印「弹射平台签名」→ 收紧
       （r1 加 min fz>50 门），继续训。
  value_loss>50 连续 5 次 → 停线（价值拟合失稳，+20 终局脉冲/Gamma 反摊失控）。
  iter<800 禁止判死刑（救援课程先行）。
- 每 200 迭代确定性评估 10 回合（固定种子）：成功率、平均最高骨盆高（progress
  监测核心——策略能否越过 0.63 的开环上限）、平均终点高、平均长度、EMA 终值。
  auto 课程下 eval 跟随当前课程起点（squat 救援期测 squat 技能、其余测 tuck）；
  iter1200 回 tuck 时加测一次 tuck 定论基线，判据以此为准（防 squat eval 数字
  污染 tuck 判据）。
- CPU 实测 10 环境 ~2100 env.step/s：2048 步/iter 收集约 1s + PPO 更新约 0.5s，
  2000 迭代 ≈ 50 分钟（8 环境顺序步进）。

用法：
  python 11_train_getup_ground.py --iters 2000                          # 地面主课（默认）
  python 11_train_getup_ground.py --iters 2000 --course auto            # 全课程 + 预登记判据
  python 11_train_getup_ground.py --iters 200 --smoke                   # 快速冒烟 + 05 的 PPO 自检
  python 11_train_getup_ground.py --eval checkpoints_getup_ground/ppo_gg_final.pt
  python 11_train_getup_ground.py --gpu
  # run 3（裁定协议）：v4d warm-start + r9 预带 + z 调度，tuck 单课 2000 iter，
  # 五分支终判 + 守卫 + squat@pitch0 侧查（预登记全文打印在日志头）
  python 11_train_getup_ground.py --warmstart checkpoints_getup/ppo_getup_v4d_best.pt \
      --iters 2000 --save-dir checkpoints_getup_ground_v3
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

import mujoco
import numpy as np
import torch

ppo_mod = importlib.import_module("05_ppo")
PPO = ppo_mod.PPO
collect_rollout_vec = ppo_mod.collect_rollout_vec
smoke_test = ppo_mod.smoke_test
gg_mod = importlib.import_module("10_get_up_ground")

OBS_DIM, ACT_DIM = 62, 12

# ---- v4d warm-start 特征映射（run 3 裁定，/tmp/gg_warmstart.py 实测零训练
#      @squat pitch-0 20/20 成功——v4d 带内技能完整转移）----
# 07 obs[45] → 10 obs[62] 的列映射：q_leg/qd_leg/prev_a(前 8)/ω/g/v/接触/h 偏移 +4。
WARM_MAP = (list(range(0, 12)) + list(range(12, 24)) + list(range(24, 32)) +
            list(range(36, 39)) + list(range(39, 42)) + list(range(42, 45)) +
            list(range(45, 47)) + list(range(47, 49)))
WARM_UNMAPPED = [c for c in range(OBS_DIM) if c not in WARM_MAP]


def build_warm_start(path: str) -> tuple[PPO, dict, dict]:
    """v4d→62/12 特征映射 warm-start（裁定协议）：
    - 特征列映射复制 v4d 权重、未映射 17 列 N(0,0.02)；
    - mean_head 前 8 行（腿）复制、后 4 行（臂）N(0,0.02)；
    - value_head.weight/bias ← 0（零重置：丢弃 07 的价值尺度，防 stale value）；
    - log_std 全 12 维 ← −1.0。
    返回 (ppo, warm_sd, backup)。backup 保存 v4d 的 log_std 逐维值与 value_head，
    供守卫回退（iter200/400 抑制 → log_std 回退；iter600 → value_head 恢复 v4d）。
    """
    sd07 = torch.load(path, map_location="cpu", weights_only=False)
    ppo = PPO(OBS_DIM, ACT_DIM, ent_coef=0.002)
    sd = ppo.ac.state_dict()
    torch.manual_seed(0)
    with torch.no_grad():
        w = torch.zeros(128, OBS_DIM)
        w[:, WARM_MAP] = sd07["trunk.0.weight"]
        w[:, WARM_UNMAPPED] = torch.randn(128, len(WARM_UNMAPPED)) * 0.02
        sd["trunk.0.weight"].copy_(w)
        sd["trunk.0.bias"].copy_(sd07["trunk.0.bias"])
        sd["trunk.2.weight"].copy_(sd07["trunk.2.weight"])
        sd["trunk.2.bias"].copy_(sd07["trunk.2.bias"])
        sd["value_head.weight"].zero_()
        sd["value_head.bias"].zero_()
        sd["mean_head.weight"][:8].copy_(sd07["mean_head.weight"])
        sd["mean_head.weight"][8:] = torch.randn(4, 128) * 0.02
        sd["mean_head.bias"][:8].copy_(sd07["mean_head.bias"])
        sd["mean_head.bias"][8:] = 0.0
        sd["log_std"].fill_(-1.0)
    ppo.ac.load_state_dict(sd)
    backup = {
        "v4d_log_std": sd07["log_std"].clone(),
        "v4d_value_head_w": sd07["value_head.weight"].clone(),
        "v4d_value_head_b": sd07["value_head.bias"].clone(),
    }
    return ppo, sd, backup


def make_env():
    return importlib.import_module("10_get_up_ground").T800GetUpGroundEnv()


def make_ppo() -> PPO:
    ppo = PPO(OBS_DIM, ACT_DIM, ent_coef=0.002)
    # 绝对目标角任务动作幅度大：初始 std 取 ~0.45，鼓励早期大范围探索目标姿态（08 同款）
    ppo.ac.log_std.data.fill_(-0.8)
    return ppo


class VecGetupEnv:
    """进程内多环境：n 个 T800 实例顺序步进，供批量采样摊薄网络开销（同 05/08 结构）。
    run 3 扩展：每环境独立 (start, pitch) 选项——守卫触发时混 20% squat@pitch-0
    保 v4d 平衡技能（starts[i]/pitches[i]）。"""

    def __init__(self, n: int, start: str = "tuck"):
        self.envs = [make_env() for _ in range(n)]
        self.n = n
        self.starts = [start] * n
        self.pitches = [None] * n          # None = 环境默认 TUCK_BASE_PITCH
        self.obs = np.stack([self._reset_one(e, i)[0] for i, e in enumerate(self.envs)]).astype(np.float32)

    def _reset_one(self, e, i: int):
        opts = {"start": self.starts[i]}
        if self.pitches[i] is not None:
            opts["pitch"] = self.pitches[i]
        return e.reset(options=opts)

    def set_start(self, start: str) -> None:
        """切换课程起点（squat 救援课 ↔ tuck 地面定论课）。"""
        self.starts = [start] * self.n
        self.pitches = [None] * self.n
        self.obs = np.stack([self._reset_one(e, i)[0] for i, e in enumerate(self.envs)]).astype(np.float32)

    def set_mix(self, squat_frac: float, squat_pitch: float = 0.0) -> None:
        """run 3 守卫：前 squat_frac 比例的环境切 squat@pitch-0，其余 tuck。
        （squat 侧查 <3/5 → 混 20% 保 v4d 技能；pitch=0 是 v4d 平衡流形）"""
        k = max(1, int(round(self.n * squat_frac)))
        for i in range(self.n):
            if i < k:
                self.starts[i] = "squat"
                self.pitches[i] = squat_pitch
            else:
                self.starts[i] = "tuck"
                self.pitches[i] = None
        self.obs = np.stack([self._reset_one(e, i)[0] for i, e in enumerate(self.envs)]).astype(np.float32)

    def step(self, actions: np.ndarray):
        obs_list, rew = [], np.zeros(self.n)
        dones = np.zeros(self.n, dtype=bool)
        for i, e in enumerate(self.envs):
            o, r, term, trunc, _ = e.step(actions[i])
            d = term or trunc
            if d:
                o, _ = self._reset_one(e, i)
            obs_list.append(o)
            rew[i] = r
            dones[i] = d
        self.obs = np.stack(obs_list).astype(np.float32)
        return self.obs, rew, dones


def _heel_geom_ids(env) -> tuple[int, int]:
    """左右足 body 下 pos.x 最小的 geom = heel 楔（探针实证 id 15/28，按 body 动态取更稳）。"""
    m = env.model
    foot_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "LINK_FOOT_L")
    foot_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "LINK_FOOT_R")
    hl = min((gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == foot_l),
             key=lambda g: m.geom_pos[g][0])
    hr = min((gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == foot_r),
             key=lambda g: m.geom_pos[g][0])
    return hl, hr


def eval_getup_ground(ppo: PPO, episodes: int = 10, seed: int = 1234, start: str = "tuck",
                      pitch: float | None = None, extended: bool = False) -> dict:
    """确定性评估：固定种子。成功率 + 骨盆高（progress 监测核心）+ EMA 终值
    （弹射签名判据：EMA∈[0.68,0.82) ∧ max_z>0.8 → 收紧门槛）。
    start 指定起点模式；pitch 显式覆盖基座俯仰（squat 侧查 pitch=0 = v4d 流形）。
    extended：追加判据④/⑤指标——heel-floor 接触回合占比、踝 pitch 力矩饱和步
    占比、平均进带累计步数（env._success_count）。"""
    env = make_env()
    hl, hr = _heel_geom_ids(env)
    succ, max_zs, end_zs, lens, rets, ema_ends = [], [], [], [], [], []
    heel_eps, sat_steps, tot_steps, band_steps = [], 0, 0, []
    opts = {"start": start}
    if pitch is not None:
        opts["pitch"] = pitch
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i, options=opts)
        ep_ret, max_z, end_z, got_success = 0.0, obs[47] + 0.25, obs[47] + 0.25, False
        ep_heel = False
        for t in range(env.max_steps):
            action = ppo.act_deterministic(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            z = obs[47] + 0.25
            max_z = max(max_z, z)
            end_z = z
            if extended:
                tot_steps += 1
                # heel-floor 接触：heel geom 与世界地板（body 0）成对接触
                for c in env.data.contact[:env.data.ncon]:
                    pair = (c.geom1, c.geom2)
                    if pair[0] in (hl, hr) and env.model.geom_bodyid[pair[1]] == 0:
                        ep_heel = True
                    elif pair[1] in (hl, hr) and env.model.geom_bodyid[pair[0]] == 0:
                        ep_heel = True
                # 踝 pitch 力矩饱和（actuator id 4/10，帽 35 N·m）
                if max(abs(env.data.actuator_force[4]), abs(env.data.actuator_force[10])) >= 34.9:
                    sat_steps += 1
            if terminated and info.get("success"):
                got_success = True
            if terminated or truncated:
                break
        succ.append(got_success)
        max_zs.append(max_z)
        end_zs.append(end_z)
        lens.append(t + 1)
        rets.append(ep_ret)
        ema_ends.append(float(env._ema_z))
        heel_eps.append(ep_heel)
        band_steps.append(float(env._success_count))
    m = {"success_rate": float(np.mean(succ)),
         "max_z_mean": float(np.mean(max_zs)),
         "max_z_min": float(np.min(max_zs)),
         "end_z_mean": float(np.mean(end_zs)),
         "len_mean": float(np.mean(lens)),
         "ret_mean": float(np.mean(rets)),
         "ema_end_mean": float(np.mean(ema_ends))}
    if extended:
        m["heel_frac"] = float(np.mean(heel_eps))
        m["ankle_sat_frac"] = sat_steps / max(tot_steps, 1)
        m["band_steps_mean"] = float(np.mean(band_steps))
    return m


def print_eval(tag: str, m: dict) -> None:
    extra = ""
    if "heel_frac" in m:
        extra = (f" | heel接触={m['heel_frac']*100:.0f}%回合 | 踝饱和={m['ankle_sat_frac']*100:.0f}%步 "
                 f"| 进带步均={m['band_steps_mean']:.0f}")
    print(f"[eval {tag}] 成功率={m['success_rate']*100:.0f}% | 平均最高骨盆高={m['max_z_mean']:.3f}m "
          f"(最差={m['max_z_min']:.3f}) | 平均终点高={m['end_z_mean']:.3f}m | "
          f"平均长度={m['len_mean']:.0f}步 | 平均回报={m['ret_mean']:.1f} | "
          f"EMA终值={m['ema_end_mean']:.3f}{extra}", flush=True)


RUN3_PRE_REG = """[RUN3 预登记] 裁定协议（终审 workflow wf_8a7bf9bc-06a，2026-11-14）：
  起点：v4d warm-start（特征映射 WARM_MAP + value_head 零重置 + log_std 全 12 维 −1.0）
  奖励：r1 记录信用 10→12；r2 手锚整项删除（几何死项）；新增 r9 预带区块
    pre_active = ema_z≥0.72 ∧ min fz>150 ∧ com_fwd≥0 ∧ pre_cnt<150；
    r9 = pre_active×[0.10+0.08·exp(−8‖g−ẑ‖²)+0.12·min(1,exp(−0.5((fwd−0.02)/0.03)²))−0.10·mean|a|]
    步上限 0.30、回合封顶 150 步（≤45）；obs 仍 62 维（64 维已否决）
  控制：QDES_MAX_RATE z 调度——z<0.80 用 0.012、z≥0.80 用 0.006（永不用 0.018）
  课程：tuck 单课 @pitch+0.15（pitch-0 tuck 是死起点，探针实证不升）；squat 侧查
    @pitch-0 每 400 iter n=5；守卫触发时混 20% squat@pitch-0 保技能
  守卫：iter200/400 eval max_z<0.60 → log_std 前 8 维回退 v4d 逐维值；iter600 仍
    <0.60 → value_head 恢复 v4d + 网络重启 warm-start；value_loss>50 连续 5 次停线
    （前 10 iter 豁免）
  终判 iter2000（基线校准 2026-11-14 实测：旧速率 0.006 下 warm-start 零训练
    max_z mean = 0.627(seed1234) / 0.663(seed7000)；本 run 新速率 0.012 实测
    iter0 基线 0.758——速率+种子效应已分解，判据阈值维持绝对值不变）：
    ① 成功率>0 → 可行，转正式验收（跨全期选点 n=50 + 行为指标 + 强制手接探针）
    ② 成功=0 ∧ max_z<0.75 → 「训练失败」签名——不是硬件墙（run-2 判②重释：从零
       学习失败；抑制签名 = trained max_z 0.558 < 未训练基线 0.666 ∧ 熵 7.6→10.4）
    ③ 成功=0 ∧ max_z>0.8 ∧ EMA终∈[0.68,0.82) → 弹射平台签名 → r1 加 min fz>50 门，
       续训至 2400
    ④ 成功=0 ∧ max_z≥0.75 ∧ EMA终<0.68 ∧ 从未 heel-floor → 「交接缺口」（79 N·m
       耦合：fz×(cop−ank)=833×0.095 >> 踝 35 帽）
    ④a heel-floor 发生过 ∧ 踝 τ 饱和 → 硬件受限交接
    ⑤ 成功=0 ∧ max_z∈[0.72,0.87) ∧ EMA终∈[0.60,0.80) ∧ 无进带尝试 → 「预带栖息」
       奖励伪影
  手接探针（判①达成后强制）：进带时刻 (qpos,qvel) 快照 → 07 env → v4d 策略 →
    07 带（h≥0.95 ∧ tilt≤0.22）150 步保持率"""


def apply_run3_verdict(it: int, m: dict, mext: dict, extended_once: bool) -> str:
    """run 3 五分支终判（预登记）。返回 'continue'/'extend'/'done'。"""
    succ = m["success_rate"] > 0.0
    maxz = m["max_z_mean"]
    ema = m["ema_end_mean"]
    if succ:
        print(f"[VERDICT① iter{it}] 成功率 {m['success_rate']*100:.0f}% > 0 → 地面站起闭环可行！"
              "转正式验收：跨全期扫 checkpoint n=50（/tmp/gg_select.py --start tuck）+ 行为指标"
              "（进带步数/mean|a|/flicker/回报护栏）+ 强制手接探针（进带快照 → 07/v4d 150 步保持率）。",
              flush=True)
        return "done"
    if maxz < 0.75:
        print(f"[VERDICT② iter{it}] 成功=0 ∧ max_z={maxz:.3f}<0.75 → 「训练失败」签名（非硬件墙）。"
              "诊断依据：warm-start 零训练已能受控升到瞬时 0.80（>开环天花板 0.60），"
              "若训练后反而 <0.75 是奖励/训练抑制，不是物理墙。run-2 判②同步重释："
              "从零学习失败（trained 0.558 < 基线 0.666 ∧ 熵 7.6→10.4），硬件墙不成立。",
              flush=True)
        return "done"
    if maxz > 0.8 and 0.68 <= ema < 0.82:
        if not extended_once:
            print(f"[VERDICT③ iter{it}] 成功=0 ∧ max_z={maxz:.3f}>0.8 ∧ EMA终={ema:.3f}∈[0.68,0.82) "
                  "→ 弹射平台签名 → r1 加 min fz>50 门（防足卸载弹射收割），续训至 2400。", flush=True)
            gg_mod.R1_GATE_MIN_FZ = 50.0
            return "extend"
        print(f"[VERDICT③ iter{it}] 二次触发（r1 门已加）→ 按 2400 终态数据归入 ②/⑤ 判。",
              flush=True)
        return "done"
    if maxz >= 0.75 and ema < 0.68:
        if mext["heel_frac"] == 0.0:
            print(f"[VERDICT④ iter{it}] 成功=0 ∧ max_z={maxz:.3f}≥0.75 ∧ EMA终={ema:.3f}<0.68 ∧ "
                  f"从未 heel-floor（{mext['heel_frac']*100:.0f}% 回合）→ 「交接缺口」："
                  "CoP 需要从足前盒角点跨 9.5cm 到足中时，踝帽 35 N·m 无法支付"
                  "fz×(cop−ank)=833×0.095≈79 N·m——受控升程卡在足跟放不下来的那一步。",
                  flush=True)
        elif mext["ankle_sat_frac"] > 0.5:
            print(f"[VERDICT④a iter{it}] 成功=0 ∧ max_z={maxz:.3f}≥0.75 ∧ EMA终={ema:.3f}<0.68 ∧ "
                  f"heel-floor 发生过（{mext['heel_frac']*100:.0f}% 回合）∧ 踝 τ 饱和"
                  f"（{mext['ankle_sat_frac']*100:.0f}% 步）→ 硬件受限交接：踝 35 N·m 帽坐实，"
                  "0.87 受控站立判不可达（改造项：踝力矩 35→150 / 踝行程 ±39°→更大 / 腰俯仰关节）。",
                  flush=True)
        else:
            print(f"[VERDICT④-数据 iter{it}] 成功=0 ∧ max_z={maxz:.3f}≥0.75 ∧ EMA终={ema:.3f}<0.68；"
                  f"heel={mext['heel_frac']*100:.0f}% 回合、踝饱和={mext['ankle_sat_frac']*100:.0f}% 步"
                  "——④/④a 子条件不完全匹配，以数据为准上报。", flush=True)
        return "done"
    if 0.72 <= maxz < 0.87 and 0.60 <= ema < 0.80 and mext["band_steps_mean"] == 0.0:
        print(f"[VERDICT⑤ iter{it}] 成功=0 ∧ max_z={maxz:.3f}∈[0.72,0.87) ∧ EMA终={ema:.3f}∈[0.60,0.80) "
              f"∧ 无进带尝试（进带步均 {mext['band_steps_mean']:.0f}）→ 「预带栖息」奖励伪影："
              "r9 在 0.72 预带区付费足够多，策略学到栖息而非过带——收紧 r9 门槛/改进度结算。",
              flush=True)
        return "done"
    print(f"[VERDICT-OTHER iter{it}] 五分支均未精确命中：成功={m['success_rate']*100:.0f}% "
          f"max_z={maxz:.3f} EMA终={ema:.3f} heel={mext['heel_frac']*100:.0f}% "
          f"踝饱和={mext['ankle_sat_frac']*100:.0f}% 进带步均={mext['band_steps_mean']:.0f}"
          "——以最近分支语义 + 数据如实上报。", flush=True)
    return "done"


def train_run3(ppo: PPO, warm_sd: dict, backup: dict, iters: int, n_steps: int,
               nenvs: int, save_dir: str, eval_every: int = 200, log_every: int = 20) -> None:
    """run 3：v4d warm-start + r9 预带奖励 + z 调度速率的 tuck 单课（裁定协议）。

    流程：iter0 基线 n=20（tuck + squat@pitch0 侧查）→ 每 200 iter n=20 eval @tuck →
    每 400 iter n=5 侧查 @squat pitch-0 → 守卫 200/400（log_std 回退）/600（value_head
    恢复+重启）→ iter2000 五分支终判（③ → r1 门 + 延至 2400）。
    """
    os.makedirs(save_dir, exist_ok=True)
    venv = VecGetupEnv(nenvs, start="tuck")
    vl_alarm = 0
    logstd_reverted = False
    extended_once = False
    total_iters = iters
    print(RUN3_PRE_REG, flush=True)
    print(f"device={ppo_mod.DEVICE}  iters={iters}  n_steps={n_steps}  nenvs={nenvs}  "
          f"course=tuck-run3  warmstart=特征映射+value_head零+log_std-1.0", flush=True)

    # ---- iter0 基线（warm-start 零训练，预期复现 max_z mean 0.663）----
    m0 = eval_getup_ground(ppo, episodes=20, start="tuck")
    print_eval("iter0[tuck]基线", m0)
    m0s = eval_getup_ground(ppo, episodes=5, start="squat", pitch=0.0)
    print_eval("iter0[squat@pitch0]基线", m0s)

    it = 0
    while it < total_iters:
        it += 1
        t0 = time.time()
        rollout, ep_rewards, ep_lens = collect_rollout_vec(venv, ppo, n_steps)
        stats = ppo.train_iteration(rollout)
        mean_ep = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        # 价值失稳停线（预登记：value_loss>50 连续 5 次；前 10 iter 豁免——warm-start
        # value_head 零重置，早期价值拟合天然大）
        if stats["value_loss"] > 50.0 and it > 10:
            vl_alarm += 1
            print(f"[ALARM] iter {it}: value_loss={stats['value_loss']:.1f}>50 "
                  f"(连续 {vl_alarm}/5)", flush=True)
            if vl_alarm >= 5:
                print("[STOP] value_loss>50 连续 5 次，停线。", flush=True)
                break
        else:
            vl_alarm = 0
        if it % log_every == 0:
            print(f"iter {it:5d} | mean_ep_return={mean_ep:8.1f} mean_len={mean_len:6.1f} | "
                  f"policy_loss={stats['policy_loss']:7.4f} value_loss={stats['value_loss']:7.4f} "
                  f"entropy={stats['entropy']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if it % eval_every == 0:
            m = eval_getup_ground(ppo, episodes=20, start="tuck", extended=True)
            print_eval(f"iter{it}[tuck]", m)
            # ---- 守卫 ----
            if it in (200, 400) and not logstd_reverted and m["max_z_mean"] < 0.60:
                with torch.no_grad():
                    ppo.ac.log_std.data[:8].copy_(backup["v4d_log_std"])
                logstd_reverted = True
                print(f"[GUARD iter{it}] 抑制签名（max_z {m['max_z_mean']:.3f}<0.60）→ "
                      f"log_std 前 8 维回退 v4d 逐维 {np.round(backup['v4d_log_std'].numpy(), 3)}",
                      flush=True)
            if it == 600 and m["max_z_mean"] < 0.60:
                sd2 = {k: v.clone() for k, v in warm_sd.items()}
                with torch.no_grad():
                    sd2["value_head.weight"].copy_(backup["v4d_value_head_w"])
                    sd2["value_head.bias"].copy_(backup["v4d_value_head_b"])
                ppo.ac.load_state_dict(sd2)
                ppo.opt = torch.optim.Adam(ppo.ac.parameters(), lr=3e-4)
                print("[GUARD iter600] max_z<0.60 持续 → value_head 恢复 v4d + 网络重启 "
                      "warm-start（优化器重建），续训", flush=True)
        if it % 400 == 0:
            ms = eval_getup_ground(ppo, episodes=5, start="squat", pitch=0.0)
            print_eval(f"iter{it}[squat@pitch0侧查]", ms)
            if ms["success_rate"] < 0.6:
                venv.set_mix(0.2, squat_pitch=0.0)
                print("[GUARD-MIX] squat@pitch0 侧查 <3/5 → 训练混 20% squat@0 保 v4d 技能",
                      flush=True)
        if it % 200 == 0:
            ppo.save(os.path.join(save_dir, f"ppo_gg_{it}.pt"))
        # ---- 终判 ----
        if it == iters:
            m_v = eval_getup_ground(ppo, episodes=20, start="tuck", extended=True)
            print_eval(f"iter{it}[tuck]终判", m_v)
            action = apply_run3_verdict(it, m_v, m_v, extended_once)
            if action == "extend":
                extended_once = True
                total_iters = 2400
                print("[RUN3] 判③触发 → r1 min fz>50 门已生效（gg_mod.R1_GATE_MIN_FZ=50.0），"
                      "训练延至 2400。", flush=True)
            elif action == "done":
                break
        if it == 2400:
            m_v = eval_getup_ground(ppo, episodes=20, start="tuck", extended=True)
            print_eval("iter2400[tuck]终判", m_v)
            apply_run3_verdict(it, m_v, m_v, extended_once)
            break
    ppo.save(os.path.join(save_dir, "ppo_gg_final.pt"))
    print("saved:", os.path.join(save_dir, "ppo_gg_final.pt"), flush=True)
    print_eval("final", eval_getup_ground(ppo, episodes=20, start="tuck", extended=True))


def train(ppo: PPO, iters: int, n_steps: int, nenvs: int, save_dir: str,
          eval_every: int = 200, log_every: int = 20, course: str = "tuck") -> None:
    os.makedirs(save_dir, exist_ok=True)
    venv = VecGetupEnv(nenvs, start="squat" if course == "squat" else "tuck")
    course_state = "squat" if course == "squat" else "tuck"
    vl_alarm = 0
    print(f"device={ppo_mod.DEVICE}  iters={iters}  n_steps={n_steps}  nenvs={nenvs}  course={course}",
          flush=True)
    for it in range(1, iters + 1):
        t0 = time.time()
        rollout, ep_rewards, ep_lens = collect_rollout_vec(venv, ppo, n_steps)
        stats = ppo.train_iteration(rollout)
        mean_ep = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        # 价值失稳停线（预登记：value_loss>50 连续 5 次 → 停线查奖励尺度）
        if stats["value_loss"] > 50.0:
            vl_alarm += 1
            print(f"[ALARM] iter {it}: value_loss={stats['value_loss']:.1f}>50 "
                  f"(连续 {vl_alarm}/5)，+20 终局脉冲/Gamma 反摊可能失控", flush=True)
            if vl_alarm >= 5:
                print("[STOP] value_loss>50 连续 5 次，停线。先查奖励尺度与 r1 记录信用，"
                      "再考虑 +20 终局 sigmoid 连续化兜底。", flush=True)
                break
        else:
            vl_alarm = 0
        if it % log_every == 0:
            print(f"iter {it:5d} | mean_ep_return={mean_ep:8.1f} mean_len={mean_len:6.1f} | "
                  f"policy_loss={stats['policy_loss']:7.4f} value_loss={stats['value_loss']:7.4f} "
                  f"entropy={stats['entropy']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if it % eval_every == 0:
            m = eval_getup_ground(ppo, start=course_state)
            print_eval(f"iter{it}[{course_state}]", m)
            m_verdict = m
            if course == "auto":
                # ---- 预登记判据（只打印/切课，不自动杀训练）----
                if it == 800 and m["success_rate"] == 0.0 and m["max_z_mean"] < 0.68:
                    print("[RESCUE iter800] 成功率 0 且 max_z<0.68 → 切 squat 救援课程 400 迭代"
                          "（补 0.69→0.87 过墙段技能，iter1200 回 tuck 定论）", flush=True)
                    venv.set_start("squat")
                    course_state = "squat"
                elif it == 1200 and course_state == "squat":
                    # 救援课结束时加测一次 tuck 定论基线（救援后的迁移探针），判据用它——
                    # 否则会把 squat 模式的 eval 数字误判到 tuck 定论上。
                    m_verdict = eval_getup_ground(ppo, start="tuck")
                    print_eval("iter1200[tuck定论]", m_verdict)
                    print("[COURSE] 救援课程结束，iter1200 回 tuck 地面定论。", flush=True)
                    venv.set_start("tuck")
                    course_state = "tuck"
                if it >= 1200 and course_state == "tuck":
                    if m_verdict["success_rate"] > 0.0:
                        print(f"[VERDICT iter{it}] 成功率>0 → 闭环可行，继续完整验收；"
                              "终选必须跨全期扫 checkpoint（MEMORY：v4d 峰值 iter1800 后退化）。",
                              flush=True)
                    elif m_verdict["max_z_mean"] > 0.8 and 0.68 <= m_verdict["ema_end_mean"] < 0.82:
                        print(f"[VERDICT iter{it}] 弹射平台签名（成功率 0 ∧ EMA∈[0.68,0.82) ∧ "
                              "max_z>0.8）→ 收紧：r1 加 min fz>50 门，继续训。", flush=True)
                    elif m_verdict["max_z_mean"] < 0.75:
                        print(f"[VERDICT iter{it}] 硬件瓶颈告警（成功率 0 ∧ max_z<0.75）→ "
                              "iter2000 复核；若维持则 0.87 受控站立判不可达，如实上报"
                              "（踝 35N·m CoP±8.4cm / 无腰俯仰 / 踝±39° / 摩擦136N<拉链300-550N）。",
                              flush=True)
        if it % 200 == 0:
            ppo.save(os.path.join(save_dir, f"ppo_gg_{it}.pt"))
    ppo.save(os.path.join(save_dir, "ppo_gg_final.pt"))
    print("saved:", os.path.join(save_dir, "ppo_gg_final.pt"), flush=True)
    print_eval("final", eval_getup_ground(ppo, start=course_state))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=2000, help="训练迭代数")
    p.add_argument("--nsteps", type=int, default=2048, help="每迭代收集步数（总环境步 = iters×nsteps）")
    p.add_argument("--nenvs", type=int, default=8, help="并行环境数")
    p.add_argument("--gpu", action="store_true", help="网络放 GPU（MuJoCo 物理仍在 CPU）")
    p.add_argument("--smoke", action="store_true", help="训练前先跑 05 的 PPO 自检")
    p.add_argument("--eval", type=str, default="", help="只评估指定 checkpoint")
    p.add_argument("--resume", type=str, default="", help="从 checkpoint 续训")
    p.add_argument("--save-dir", type=str, default="checkpoints_getup_ground",
                   help="checkpoint 保存目录")
    p.add_argument("--warmstart", type=str, default="",
                   help="run 3：从 07 v4d checkpoint 特征映射 warm-start（自动启用 run-3 协议）")
    p.add_argument("--course", type=str, default="tuck", choices=["tuck", "squat", "auto"],
                   help="课程表：tuck 地面主课（默认）/ squat 带内技能课 / auto 全课程+预登记判据")
    p.add_argument("--start", type=str, default="tuck", choices=["tuck", "squat"],
                   help="--eval 单评估时的起点模式（救援期 checkpoint 侧查 squat 技能用）")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu:
        ppo_mod.DEVICE = torch.device("cuda")
        print("GPU 训练：网络在 cuda，物理仍在 CPU", flush=True)

    if args.smoke:
        smoke_test()

    if args.warmstart:
        # ---- run 3（裁定协议）：warm-start + r9 预带奖励 + z 调度，tuck 单课 ----
        ppo, warm_sd, backup = build_warm_start(args.warmstart)
        print(f"warm-start 已装入：{args.warmstart}（value_head 零重置、log_std 全 12 维 −1.0）",
              flush=True)
        train_run3(ppo, warm_sd, backup, args.iters, args.nsteps, args.nenvs, args.save_dir)
        return

    ppo = make_ppo()
    if args.eval:
        ppo.load(args.eval)
        print(f"已加载策略：{args.eval}")
        print_eval("loaded", eval_getup_ground(ppo, episodes=20, start=args.start))
        return
    if args.resume:
        ppo.load(args.resume)
        print(f"续训起点：{args.resume}")

    train(ppo, args.iters, args.nsteps, args.nenvs, args.save_dir, course=args.course)


if __name__ == "__main__":
    main()
