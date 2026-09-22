"""Build the whole-scene mesh cache (data/scene_cache/<task>/) for every task.

Usage: .venv/bin/python scripts/build_scene_cache.py [--task Task1_Kitchen_Cleanup]
"""
import argparse
import time
from pathlib import Path

from realmirror_annot.scene import SceneCacheBuilder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--out", default="data/scene_cache")
    args = ap.parse_args()
    b = SceneCacheBuilder(root=Path(__file__).resolve().parents[1])
    for task in b.scene_cfg["tasks"]:
        if args.task and task != args.task:
            continue
        t0 = time.time()
        objs = b.build(task, args.out)
        nb = sum(o.background for o in objs)
        nf = sum(o.num_faces for o in objs)
        print(f"{task}: {len(objs)} objects ({nb} background, {sum(o.frame == 'tracked' for o in objs)} tracked), faces {nf:,} (from {sum(o.num_faces_original for o in objs):,}), {time.time() - t0:.0f}s")
        for o in objs:
            print(f"   {o.obj_id:3d} {o.name:28s} {o.category:18s} {o.frame:7s} faces={o.num_faces:>7,} {o.prim_path}")


if __name__ == "__main__":
    main()
