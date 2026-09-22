"""Annotate one RealMirror evaluation episode.

Usage:
  .venv/bin/python scripts/annotate_episode.py --episode <dir with trajectory.hdf5> --out outputs/<...> \
      [--render] [--review-every 10] [--video] [--max-steps N]
--render needs a GPU with EGL (PYOPENGL_PLATFORM=egl is set automatically).
"""
import argparse
import time
from pathlib import Path

from realmirror_annot.annotate import Annotator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", action="store_true", help="render instance-ID masks of the whole scene (GPU)")
    ap.add_argument("--review-every", type=int, default=0)
    ap.add_argument("--review-count", type=int, default=0, help="N evenly spaced review images per episode")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    ann = Annotator(root=Path(__file__).resolve().parents[1], render=args.render)
    steps = range(args.max_steps) if args.max_steps else None
    out = ann.annotate_episode(args.episode, args.out, review_every=args.review_every, review_video=args.video, steps=steps, review_count=args.review_count)
    print(f"wrote {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
