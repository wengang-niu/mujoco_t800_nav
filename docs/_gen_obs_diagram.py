"""生成 T800 观测空间 1083 = 15×72 + 3 的结构示意图 (PNG)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# ---- 中文字体 ----
FONT_REG = "/home/root123/.local/share/fonts/fandol/FandolHei-Regular.otf"
FONT_BOLD = "/home/root123/.local/share/fonts/fandol/FandolHei-Bold.otf"
for _p in (FONT_REG, FONT_BOLD):
    fm.fontManager.addfont(_p)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_REG).get_name()
plt.rcParams["axes.unicode_minus"] = False

# ---- 配色（5 段观测 + 指令）----
C_POS = "#4E79A7"    # 位置偏差
C_VEL = "#59A14F"    # 速度
C_ACT = "#F28E2B"    # 上一动作
C_ANG = "#B07AA1"    # 角速度
C_GRV = "#E15759"    # 投影重力
C_CMD = "#76B7B2"    # 指令
C_FRM = "#C9D2DA"    # 顶层帧块（灰）
C_DARK = "#1F2937"
C_GRAY = "#6B7280"

SEGS = [
    ("关节位置偏差", "q − default_q", 22, C_POS),
    ("关节速度", "qd", 22, C_VEL),
    ("上一动作", "prev_action", 22, C_ACT),
    ("身体角速度", "angvel", 3, C_ANG),
    ("投影重力", "proj_g", 3, C_GRV),
]

fig, ax = plt.subplots(figsize=(16, 10.5), dpi=160)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def box(x, y, w, h, fc, ec="none", r=0.6, lw=0, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha)
    ax.add_patch(p)

def txt(x, y, s, size=11, color=C_DARK, ha="center", va="center", weight="normal", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, fontstyle=style, zorder=5)

# ============ 标题 ============
txt(50, 96.5, "T800 观测空间 · 1083 = 15 帧 × 72 维 + 3 维指令", size=20, weight="bold")
txt(50, 92.5, "一个一维向量喂给策略网络：前 1080 维是 15 帧历史，末尾 3 维是运动指令",
    size=12.5, color=C_GRAY)

# ============ Row A：顶层总览 ============
txt(6, 86.5, "① 顶层布局（1083 维，内存顺序）", size=14, weight="bold", ha="left")

FW, FGAP = 4.9, 0.55
X0 = 7.0
BY, BH = 78.5, 6.0
for i in range(15):
    x = X0 + i * (FW + FGAP)
    box(x, BY, FW, BH, C_FRM)
    txt(x + FW / 2, BY + BH / 2, f"f{i}", size=11, weight="bold", color=C_DARK)
    txt(x + FW / 2, BY - 1.6, "72", size=9.5, color=C_GRAY)

# 指令块
CX = X0 + 15 * (FW + FGAP) + 0.6
box(CX, BY, 6.0, BH, C_CMD)
txt(CX + 3.0, BY + BH / 2, "指令", size=11, weight="bold", color="white")
txt(CX + 3.0, BY - 1.6, "3", size=9.5, color=C_GRAY)

# 大括号：历史
f_end = X0 + 14 * (FW + FGAP) + FW
ax.plot([X0, X0], [BY - 3.2, BY - 4.2], color=C_GRAY, lw=1.3)
ax.plot([f_end, f_end], [BY - 3.2, BY - 4.2], color=C_GRAY, lw=1.3)
ax.plot([X0, f_end], [BY - 4.2, BY - 4.2], color=C_GRAY, lw=1.3)
txt((X0 + f_end) / 2, BY - 5.2, "15 × 72 = 1080 维（历史）", size=11.5, color=C_DARK, weight="bold")
# 指令小括号
ax.plot([CX, CX], [BY - 3.2, BY - 4.2], color=C_GRAY, lw=1.3)
ax.plot([CX + 6, CX + 6], [BY - 3.2, BY - 4.2], color=C_GRAY, lw=1.3)
ax.plot([CX, CX + 6], [BY - 4.2, BY - 4.2], color=C_GRAY, lw=1.3)
txt(CX + 3.0, BY - 5.2, "3 维指令\n[vx, vy, yaw]", size=10, color=C_DARK, weight="bold")

# ============ Row B：单帧放大 ============
txt(6, 68.5, "② 放大其中一帧 frame k（占索引 [72k … 72k+71]）", size=14, weight="bold", ha="left")

scale = 1.55   # 每维的宽度单位
ZY = 58.0      # 段 bar 的 y
ZH = 5.2
segs_x = 0.0
seg_w = []
for name, sym, n, c in SEGS:
    seg_w.append(n * scale)
tot = sum(seg_w)
sx = (100 - tot) / 2
cur = sx
for (name, sym, n, c), w in zip(SEGS, seg_w):
    box(cur, ZY, w, ZH, c, r=0.5)
    # 名称
    txt(cur + w / 2, ZY + ZH + 1.9, name, size=11.5, weight="bold", color=c)
    txt(cur + w / 2, ZY + ZH + 0.45, sym, size=9, color=C_GRAY)
    # 维度数（大）
    txt(cur + w / 2, ZY + ZH / 2, f"{n}", size=13, weight="bold", color="white")
    cur += w

# 索引刻度（帧内累计偏移）
off = 0
cur = sx
for (name, sym, n, c), w in zip(SEGS, seg_w):
    txt(cur, ZY - 2.0, f"{off}", size=8, color=C_GRAY, ha="center")
    txt(cur + w, ZY - 2.0, f"{off + n - 1}", size=8, color=C_GRAY, ha="center")
    txt(cur + w / 2, ZY - 3.4, f"{n} 维", size=9, color=C_GRAY)
    off += n
    cur += w
txt(sx, ZY - 5.0, f"帧内偏移 0 … 71（对应全局索引 72k … 72k+71）", size=10, color=C_DARK, ha="left")

# 连接箭头：顶层 -> 放大
ax.annotate("", xy=(sx + tot / 2, ZY + ZH + 0.3), xytext=(X0 + 2.45, BY),
            arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.4,
                            connectionstyle="arc3,rad=-0.15"))
txt(28, 71.2, "任取 1 帧放大", size=9.5, color=C_GRAY, style="italic")

# ============ Row C：滑动窗口 ============
txt(6, 46.0, "③ 历史是 FIFO 队列：每 0.01s 新帧进、最老帧出", size=14, weight="bold", ha="left")

SW = 4.9
rows = [
    ("t = 0.00s", [f"f{i}" for i in range(15)], None),
    ("t = 0.01s", [f"f{i}" for i in range(1, 16)], "f15"),
    ("t = 0.02s", [f"f{i}" for i in range(2, 17)], "f16"),
]
ry = 36.0
for label, frames, newf in rows:
    txt(6, ry + 1.6, label, size=11, weight="bold", ha="left", color=C_DARK)
    x = 24.0
    for j, f in enumerate(frames):
        is_new = (newf is not None and j == len(frames) - 1)
        box(x, ry, SW, 3.6, C_CMD if is_new else C_FRM, r=0.4)
        txt(x + SW / 2, ry + 1.8, f, size=9.5, weight="bold",
            color="white" if is_new else C_DARK)
        if is_new:
            txt(x + SW / 2, ry - 1.5, "新进", size=8.5, color=C_CMD, weight="bold")
        x += SW + 0.45
    # 说明
    if newf is not None:
        txt(x + 1.2, ry + 1.8, "← 新帧进入，最老帧被挤出", size=9.5, color=C_GRAY, ha="left")
    ry -= 8.4

# ============ 图例 ============
txt(6, 12.5, "图例", size=12, weight="bold", ha="left")
lx = 6.0
for name, sym, n, c in SEGS:
    box(lx, 9.2, 2.0, 2.0, c, r=0.3)
    txt(lx + 3.6, 10.2, f"{name} ({n} 维)", size=10.5, ha="left", color=C_DARK)
    lx += 17.0
box(lx, 9.2, 2.0, 2.0, C_CMD, r=0.3)
txt(lx + 3.6, 10.2, "指令 command (3 维)", size=10.5, ha="left", color=C_DARK)

txt(50, 5.2, "1083 = (22+22+22+3+3) × 15 + 3 = 72 × 15 + 3 = 1080 + 3",
    size=14, weight="bold", color=C_DARK)

plt.tight_layout(pad=0.5)
out = "/home/root123/workspace/source/mujoco_t800_nav/docs/observation_1083.png"
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("saved:", out)
