"""为 mujoco/scenes/ 下的场景拉取共享家具资源（thor + objaverse）。

mujoco/objects/ 已被 .gitignore 忽略（约 2.5GB），用本脚本重建：

    python tools/fetch_scene_assets.py cafe_37874

原理：molmospaces 数据集用 parquet 索引记录每个资源包在 2GB shard tar 里的
byte offset/size，脚本发 HTTP Range 请求只拉所需片段 → zstd 解压 → 解包到
mujoco/objects/{thor,objaverse}。XML 里的 `../../objects/...` 相对路径正好
解析到 mujoco/objects/，无需改 MJCF。

不下载 Textures / SkyBox（+868MB，仅 11 个 PNG 被引用，而加载的
train_37874_stripped.xml 不引用它们，无需下载）。
"""
from __future__ import annotations

import concurrent.futures
import gzip
import io
import json
import os
import re
import sys
import tarfile
import urllib.request

import pandas as pd
import zstandard

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # tools/ 的上级 = 仓库根
SCENES = os.path.join(REPO, "mujoco", "scenes")
OBJECTS = os.path.join(REPO, "mujoco", "objects")

THOR_PARQUET = (
    "https://huggingface.co/datasets/allenai/molmospaces/resolve/main/"
    "mujoco__objects__thor__20251117/pkgs-00000-of-00001.parquet"
)
THOR_SHARD = (
    "https://huggingface.co/datasets/allenai/molmospaces/resolve/main/"
    "mujoco/objects/thor/{ver}/shards/{s:05d}.tar"
)
THOR_VER = "20251117"  # 与 parquet 同版本目录，失败时下面会探测

OBJ_PARQUET = (
    "https://huggingface.co/datasets/allenai/molmospaces/resolve/main/"
    "mujoco/objects/objaverse/20260131/pkgs-00000-of-00001.parquet"
)
OBJ_META = (
    "https://huggingface.co/datasets/allenai/molmospaces/resolve/main/"
    "mujoco/objects/objaverse/20260131/mjthor_resources_combined_meta.json.gz"
)
OBJ_SHARD = (
    "https://huggingface.co/datasets/allenai/molmospaces/resolve/main/"
    "mujoco/objects/objaverse/20260131/shards/{s:05d}.tar"
)

SKIP_CATS = {"Textures", "SkyBox"}
THOR_BASE = "https://huggingface.co/datasets/allenai/molmospaces/resolve/main/mujoco/objects/thor"


def range_fetch(url: str, off: int, size: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={off}-{off + size - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def zstd_decompress(blob: bytes) -> bytes:
    return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(blob)).read()


def scene_external_paths(scene_dir: str, xml_name: str) -> tuple[set[str], set[str]]:
    """返回 (thor 相对路径集合, objaverse 对象 id 集合)"""
    text = open(os.path.join(scene_dir, xml_name), encoding="utf-8").read()
    thor = set(re.findall(r'\.\./\.\./objects/thor/([^"]+)', text))
    thor = {p for p in thor if p.split("/")[0] not in SKIP_CATS}
    obj_ids = set(re.findall(r"objects/objaverse/([0-9a-f]{32})", text))
    return thor, obj_ids


def fetch_thor(cats: set[str]) -> None:
    df = pd.read_parquet(THOR_PARQUET)
    # 分类名 → 包名: path 形如 thor_Kitchen Objects.tar.zst
    want = []
    for _, row in df.iterrows():
        p = row["path"]
        if not p.startswith("thor_") or not p.endswith(".tar.zst"):
            continue
        # SkyBox 特殊
        if p == "thor_SkyBox/Sky.tar.zst":
            cat = "SkyBox"
        else:
            cat = p[len("thor_"):-len(".tar.zst")]
        if cat in cats and cat not in SKIP_CATS:
            want.append((cat, row))
    # 版本目录探测：用第一个包试 20251117
    ver = "20251117"
    # parquet 来自 20251117 config，shard 路径
    def dl(item):
        cat, row = item
        url = THOR_SHARD.format(ver=ver, s=int(row["shard_id"]))
        # 兼容：若 20251117 目录不存在，试其它
        try:
            blob = range_fetch(url, int(row["offset"]), int(row["size"]))
        except Exception:
            # 尝试 repo 根下 objects 布局
            alt = (
                "https://huggingface.co/datasets/allenai/molmospaces/resolve/main/"
                f"mujoco/objects/thor/{ver}/shards/{int(row['shard_id']):05d}.tar"
            )
            blob = range_fetch(alt, int(row["offset"]), int(row["size"]))
        data = zstd_decompress(blob)
        dest = os.path.join(OBJECTS, "thor")
        os.makedirs(dest, exist_ok=True)
        # 检查内部顶层
        tf = tarfile.open(fileobj=io.BytesIO(data))
        names = tf.getnames()
        # 若顶层已是分类目录名，直接解压；否则包一层
        tf.extractall(dest)
        return cat, len(names), sum(m.size for m in tf.getmembers())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for cat, n, sz in ex.map(dl, want):
            print(f"thor[{cat}]: {n} files, {sz/1e6:.1f} MB inflated")


def fetch_objaverse(obj_ids: set[str]) -> None:
    meta = json.load(gzip.open(io.BytesIO(
        urllib.request.urlopen(OBJ_META, timeout=120).read())))
    id2pkg: dict[str, str] = {}
    for pkg, files in meta.items():
        for f in files:
            for i in obj_ids:
                if i in f:
                    id2pkg[i] = pkg
    pkgs = sorted(set(id2pkg.values()))
    print(f"objaverse: {len(obj_ids)} ids -> {len(pkgs)} packages")
    df = pd.read_parquet(OBJ_PARQUET)
    df = df[df.path.isin(pkgs)]
    dest = os.path.join(OBJECTS, "objaverse")
    os.makedirs(dest, exist_ok=True)

    def dl(row):
        url = OBJ_SHARD.format(s=int(row["shard_id"]))
        blob = range_fetch(url, int(row["offset"]), int(row["size"]))
        data = zstd_decompress(blob)
        tf = tarfile.open(fileobj=io.BytesIO(data))
        tf.extractall(dest)
        return row["path"], len(tf.getnames())

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for path, n in ex.map(dl, [r for _, r in df.iterrows()]):
            done += 1
            if done % 20 == 0 or done == len(df):
                print(f"objaverse {done}/{len(df)} ({path}, {n} files)")


def main() -> None:
    scenes = sys.argv[1:] or ["cafe_37874"]
    thor_cats: set[str] = set()
    obj_ids: set[str] = set()
    for name in scenes:
        sdir = os.path.join(SCENES, name)
        xmls = [f for f in os.listdir(sdir)
                if f.endswith(".xml") and "ceiling" not in f]
        t, o = scene_external_paths(sdir, xmls[0])
        print(f"{name}: thor={len(t)} paths, objaverse={len(o)} ids")
        # 路径第一段是分类
        for p in t:
            thor_cats.add(p.split("/")[0])
        obj_ids |= o
    print("thor categories:", sorted(thor_cats))
    if thor_cats:
        fetch_thor(thor_cats)
    if obj_ids:
        fetch_objaverse(obj_ids)
    print("done ->", OBJECTS)


if __name__ == "__main__":
    main()
