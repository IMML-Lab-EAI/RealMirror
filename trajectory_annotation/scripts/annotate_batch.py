"""Annotate many RealMirror evaluation episodes (one process, sequential; run
several PBS jobs with --shard i/n to parallelise).

Usage:
  python scripts/annotate_batch.py --runs <eval_all_smolvla/<timestamp>> --out outputs/realmirror_smolvla \
      [--tasks Task1_Kitchen_Cleanup,...] [--render] [--shard 0/4] [--review-every 50] [--only-success]
Episodes already having a complete annotations.jsonl (+ masks.h5 when rendering) are skipped.
"""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

from realmirror_annot.annotate import Annotator


def list_episodes(runs: Path, tasks):
    eps = []
    for task_dir in sorted(runs.iterdir()):
        if not task_dir.is_dir() or task_dir.name == "logs":
            continue
        if tasks and task_dir.name not in tasks:
            continue
        for run in sorted(task_dir.iterdir()):
            traj = run / "trajectories"
            if not traj.is_dir():
                continue
            for ep in sorted(traj.glob("episode-*")):
                if (ep / "trajectory.hdf5").exists() and (ep / "metadata.json").exists():
                    eps.append((task_dir.name, run.name, ep))
    return eps


def is_done(out_dir: Path, render: bool, num_steps: int) -> bool:
    f = out_dir / "annotations.jsonl"
    if not f.exists() or (render and not (out_dir / "masks.h5").exists()):
        return False
    try:
        with open(f) as fh:
            n = sum(1 for _ in fh)
        return n >= num_steps
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th episode starting at i")
    ap.add_argument("--review-every", type=int, default=0)
    ap.add_argument("--review-count", type=int, default=3)
    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reverse", action="store_true", help="process the episode list from the end (to add workers next to a running forward job)")
    ap.add_argument("--skip-existing-dir", action="store_true", help="skip episodes whose output dir already exists (another job started them)")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = set(args.tasks.split(",")) if args.tasks else None
    i, n = (int(x) for x in args.shard.split("/"))
    eps = list_episodes(Path(args.runs), tasks)[i::n]
    if args.reverse:
        eps = eps[::-1]
    if args.limit:
        eps = eps[: args.limit]
    print(f"{len(eps)} episodes in shard {args.shard}", flush=True)
    ann = Annotator(root=root, render=args.render)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    log = open(out_root / f"batch_log_{i}of{n}.csv", "a")
    w = csv.writer(log)
    done = skipped = failed = 0
    for task, run, ep in eps:
        meta = json.loads((ep / "metadata.json").read_text())
        if args.only_success and not meta.get("success"):
            continue
        out_dir = out_root / task / run / ep.name
        if is_done(out_dir, args.render, int(meta.get("num_steps", 0))) or (args.skip_existing_dir and out_dir.exists()):
            skipped += 1
            continue
        t0 = time.time()
        try:
            ann.annotate_episode(ep, out_dir, review_every=args.review_every, review_count=args.review_count)
            status = "ok"
            done += 1
        except Exception as e:  # keep going, record the failure
            status = f"error: {e!r}"
            failed += 1
            traceback.print_exc()
        dt = time.time() - t0
        w.writerow([task, run, ep.name, meta.get("num_steps"), meta.get("success"), status, round(dt, 1)])
        log.flush()
        print(f"[{done + skipped + failed}/{len(eps)}] {task}/{ep.name} steps={meta.get('num_steps')} {status} {dt:.1f}s", flush=True)
    print(f"done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
