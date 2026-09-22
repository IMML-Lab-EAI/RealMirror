"""Extract object bounds for every RealMirror task from the downloaded scene USDs.

Usage: .venv/bin/python scripts/build_object_tables.py [--asset data/realmirror-asset] [--out data/object_tables]
"""
import argparse
import json
from pathlib import Path

import yaml

import numpy as np

from realmirror_annot.extents import build_object_points, build_object_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/realmirror_objects.yaml")
    ap.add_argument("--asset", default="data/realmirror-asset")
    ap.add_argument("--out", default="data/object_tables")
    ap.add_argument("--task", default=None, help="only this task")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task, usd_name in cfg["scene_usd"].items():
        if args.task and task != args.task:
            continue
        usd = Path(args.asset) / "scenes" / task / usd_name
        prims = list(cfg["objects"][task].keys())
        table = build_object_table(usd, prims)
        for pp, meta in cfg["objects"][task].items():
            table[pp].update(meta)
        (out / f"{task}.json").write_text(json.dumps(table, indent=2))
        pts = build_object_points(usd, prims)
        np.savez_compressed(out / f"{task}_points.npz", **{k.replace("/", "|"): v for k, v in pts.items()})
        bad = [p for p, e in table.items() if "error" in e]
        print(f"{task}: {len(table) - len(bad)} objects ok, errors={bad}")
        for pp, e in table.items():
            if "error" not in e:
                print(f"   {e['name']:28s} size_m={[round(s, 3) for s in e['size_m']]} points={len(pts.get(pp, []))}")


if __name__ == "__main__":
    main()
