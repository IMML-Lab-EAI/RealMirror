"""Apply current configs/scene_objects.yaml names (by prim_path) to finished outputs:
episode_meta.json + annotations.jsonl. Usage: python scripts/rename_in_outputs.py <out_root> [Task]"""
import json, sys
from pathlib import Path
import yaml
root = Path(sys.argv[1]); only = sys.argv[2] if len(sys.argv) > 2 else None
cfg = yaml.safe_load(open("configs/scene_objects.yaml"))["tasks"]
n = 0
for task, tcfg in cfg.items():
    if only and task != only: continue
    ren = {p: l["name"] for p, l in (tcfg.get("labels") or {}).items()}
    for d in sorted((root / task).glob("*/episode-*")):
        mp, ap = d / "episode_meta.json", d / "annotations.jsonl"
        if not mp.exists() or not ap.exists(): continue
        meta = json.loads(mp.read_text()); ch = False
        for so in meta.get("scene_objects", []):
            if so["prim_path"] in ren and so["name"] != ren[so["prim_path"]]: so["name"] = ren[so["prim_path"]]; ch = True
        if not ch: continue
        mp.write_text(json.dumps(meta, indent=2))
        tmp = d / "annotations.jsonl.tmp"
        with open(ap) as fi, open(tmp, "w") as fo:
            for line in fi:
                r = json.loads(line)
                for so in r.get("scene_objects", []):
                    if so["prim_path"] in ren: so["name"] = ren[so["prim_path"]]
                fo.write(json.dumps(r) + "\n")
        tmp.replace(ap); n += 1
        print("renamed", task, d.name, flush=True)
print("episodes renamed:", n)
