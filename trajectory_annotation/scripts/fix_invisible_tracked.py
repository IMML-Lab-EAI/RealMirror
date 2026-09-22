"""Post-fix: tracked objects that have no renderable mesh (mask_id null while
masks.h5 exists) were marked visible by frustum only. Rewrite them as
visible=false / visibility_source=invisible_in_sim and redraw review images."""
import json, sys
from pathlib import Path

import cv2
import h5py
import numpy as np

from realmirror_annot.episode import Episode
from realmirror_annot.viz import draw_step

root = Path(sys.argv[1])
fixed_eps = 0
for ann in sorted(root.glob("*/*/episode-*/annotations.jsonl")):
    d = ann.parent
    if not (d / "masks.h5").exists():
        continue
    lines = ann.read_text().splitlines()
    recs = [json.loads(l) for l in lines]
    if not any(o.get("mask_id") is None and o.get("active") for r in recs for o in r["objects"]):
        continue
    for r in recs:
        for o in r["objects"]:
            if o.get("mask_id") is None and o.get("active"):
                o["visible_in_sim"] = False
                for v in o["views"].values():
                    v["visible"] = False
                    v["visible_px"] = 0
                    v["visibility_source"] = "invisible_in_sim"
    tmp = d / "annotations.jsonl.tmp"
    with open(tmp, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(ann)
    meta = json.load(open(d / "episode_meta.json"))
    review = d / "review"
    if review.exists():
        ep = Episode(meta["episode_dir"])
        with h5py.File(d / "masks.h5", "r") as h:
            for img in sorted(review.glob("step_*.jpg")):
                t = int(img.stem.split("_")[1])
                ids = {n: h["id_map"][t, i].astype(np.int32) for i, n in enumerate(meta["cameras"])}
                canvas = draw_step({n: ep.image(n, t) for n in meta["cameras"]}, recs[t], meta, id_maps=ids)
                cv2.imwrite(str(img), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        ep.close()
    fixed_eps += 1
    print("fixed", d, flush=True)
print("episodes fixed:", fixed_eps)
