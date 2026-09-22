"""Redraw review images of finished episodes: delete old ones, draw N evenly spaced steps (no GPU)."""
import json, sys
from pathlib import Path
import cv2, h5py, numpy as np
from realmirror_annot.annotate import review_steps_for
from realmirror_annot.episode import Episode
from realmirror_annot.viz import draw_step

root = Path(sys.argv[1]); count = int(sys.argv[2]) if len(sys.argv) > 2 else 3; n = 0
task = sys.argv[3] if len(sys.argv) > 3 else "*"
for ann in sorted(root.glob(f"{task}/*/episode-*/annotations.jsonl")):
    d = ann.parent
    meta = json.load(open(d / "episode_meta.json"))
    recs = [json.loads(l) for l in open(ann)]
    if len(recs) < meta["num_steps"] or not (d / "masks.h5").exists():
        continue
    try:
        h = h5py.File(d / "masks.h5", "r")
    except OSError:
        print("corrupt masks.h5, skipped", d, flush=True); continue
    review = d / "review"; review.mkdir(exist_ok=True)
    for old in review.glob("*.jpg"): old.unlink()
    ep = Episode(meta["episode_dir"])
    for t in review_steps_for(len(recs), count):
        ids = {c: h["id_map"][t, i].astype(np.int32) for i, c in enumerate(meta["cameras"])}
        cv2.imwrite(str(review / f"step_{t:06d}.jpg"), draw_step({c: ep.image(c, t) for c in meta["cameras"]}, recs[t], meta, id_maps=ids), [cv2.IMWRITE_JPEG_QUALITY, 90])
        n += 1
    h.close(); ep.close()
    print("redrawn", d.name, flush=True)
print("images drawn:", n)
