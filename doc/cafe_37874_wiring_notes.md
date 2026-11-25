# cafe_37874 接线备忘

场景已放在 `mujoco/scenes/cafe_37874/`，资源在 `mujoco/objects/`（gitignore）。
以下是把它与 T800 合成一个模型时的实测结论（mujoco 4.0.0），避免重复踩坑。

## 1. include 顺序必须「机器人在前」

`source/walker.py` 假设自由基座在 `qpos[0:7]`（`qpos[2]` = base_z，`qpos[3:7]` = 四元数），
关节名 `J00..J24` 在 `qpos[7..31]`。

场景有 130 个关节（`nq=502`）。**若场景在前**，robot 的 freejoint 落到 `qposadr=502`，
walker.py 会静默写坏场景的门关节。**robot 在前**则 freejoint `qposadr=0`，符合假设。

```xml
<mujoco model="t800_cafe_37874">
    <include file="robot/t800/xml/serial_t800.xml"/>              <!-- 必须第一 -->
    <include file="scenes/cafe_37874/train_37874_stripped.xml"/>   <!-- 第二 -->
</mujoco>
```

## 2. 不要 include `environment/ground.xml`

场景自带 `<geom name="floor" .../>`，而 ground.xml 也有同名 geom。
**重名是硬错误**（`XML Error: repeated name 'floor' in geom`），模型直接加载失败。
场景自己的 floor plane 就是地面，同时也避免双地板 z-fighting。

## 3. 出生点：世界原点在墙里

实测 robot（walker.reset 状态，z=1.02，default_q 站姿）在 (0,0) 有 **159 个穿透接触**
（墙 `wall_2_12/2_13/2_48/2_49` + 一个货架）。

房间分布（MuJoCo 坐标，x/y 平面）：

| 房间 | 范围 |
|---|---|
| `room_1` sales floor 主厅 | x[0,8] y[3,11] |
| `room_5` bakery kitchen | x[8,12] y[7,11] |
| `room_6` back office | x[8,12] y[3,7] |
| `room_0` changing room | x[0,4] y[11,14] |
| `room_2` / `room_7` | x[0,4] / x[4,8] y[0,3] |
| `room_3` / `room_4` | x[4,7] / x[7,10] y[11,14] |

**推荐出生点：(4.0, 7.0)，yaw 0** —— sales floor 正中，实测零穿透，0.5m 网格四周净空。

移出生点的方式 —— **注意：给场景 worldbody 包一层 `<body>` 是错的**，实测报
`ValueError: free joint can only be used on top level`：场景里有 freejoint 动态物体
（objaverse 家具），MuJoCo 规定 freejoint 只能挂在 worldbody 直接子级，包一层就破了。

正确做法（已实装）：**在 `Walker.reset()` 里设 qpos**，出生点进 config：

- `config/t800.yaml` → `reset.base_pos: [4.0, 7.0]`
- `source/config.py` → `Config.base_pos`（用 `reset.get("base_pos", (0,0))` 解析，旧配置不报错）
- `source/walker.py` `reset()` → `self.data.qpos[0:2] = cfg.base_pos`

这样 robot XML 和场景 XML 都不用动，出生点是每个场景自己的配置项。

其它可行方案（不推荐）：
- 改 `serial_t800.xml` 的 `LINK_BASE pos="4 7 1.03"` —— 把场景信息耦合进机器人文件
- 给 worldbody 下**每一个**直接子 body 单独加 `pos` 偏移（含 freejoint 物体）—— 机械但要
  处理已有 `pos` 的叠加，容易错

## 4. 全局设置合并是良性的

重复的 `<option>` / `<compiler>` / `<size>` / `<visual>` **按属性合并，后出现的覆盖先出现的同名属性**。

实测合并结果：`timestep=0.002`（robot）、`integrator=implicitfast`（双方一致）、
`cone=elliptic` / `jacobian=sparse` / `noslip_iterations=4`（场景）、`njmax/nconmax=5000`（场景）、
`discardvisual=false`（robot —— 关键，保住场景约 4.7k 个视觉 geom）。

**不要删 robot 的 `<compiler>` 标签**，否则视觉 geom 会被丢弃。

## 5. 合并模型实测数据

| 项 | 值 |
|---|---|
| nq / nv | 534 / 471 |
| nbody / ngeom | 558 / 4818 |
| nu（motor_*） | 25 |
| nsensor（含 imu_angular_velocity） | 15 |

robot 的 `biped/...` class 与场景的 `__VISUAL_MJT__` 等 class 不冲突。

## 6. 性能

4818 个 geom + 场景的 `noslip_iterations=4` / `energy=enable` 让 `mj_step` 比平地场景重。
场景默认 `shadowsize=16384` 很吃力，可在顶层加 `<visual><quality shadowsize="4096"/></visual>` 覆盖。
更激进可在 includes 之后加 `<option><flag energy="disable"/></option>` 并降低 `noslip_iterations`。

## 7. 文件选择

- **`train_37874_stripped.xml`** —— 当前唯一加载的场景文件，可直接加载。
  相对原始主文件删减了 11 个缺失纹理（10 个 `thor/Textures/*.png` + 1 个
  `thor/SkyBox/SkySouthLakeUnion.png`）及其派生的 46 个关节（nbody 528 / ngeom 4764）。
- 天花板变体 `train_37874_ceiling.xml` 及其 `ceiling_*.obj` 已删除（未接线、无人引用），
  原始主文件 `train_37874.xml` 也已删除（依赖的 Textures/SkyBox 不下载，无法加载）。
  需要时均可从 molmospaces 重新拉取。
- 注意：生成 `_stripped.xml` 的工具已随 `try/` 删除，该文件现在是**冻结产物**。
  若将来补全纹理，可从 molmospaces 拉回原始主文件。

## 8. 其它

- `<statistic center extent>` 可在顶层设，用于框定 viewer 初始机位；位移后大厅中心是世界 (0,0)。
- 可加 builtin 渐变 skybox 替代缺失的 SkyBox PNG：
  ```xml
  <asset><texture name="sky_builtin" type="skybox" builtin="gradient"
           rgb1="0.55 0.65 0.75" rgb2="0.93 0.93 0.95" width="256" height="256"/></asset>
  ```
- 场景资源可重建：`python tools/fetch_scene_assets.py cafe_37874`
- 场景来源：allenai/molmospaces（holodeck-objaverse-train，`train_37874`）
