#!/usr/bin/env python3
"""MVTrack External Python Orchestrator."""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import hashlib
import random
from typing import Any, Dict, List, Tuple
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "output"
WRAPPER = str(ROOT / "run_ue5_worker.sh")


def get_output_dir(cfg: Dict) -> pathlib.Path:
    """Resolve the dataset output root from config, falling back to output/."""
    configured = cfg.get("output", {}).get("root")
    if not configured:
        return OUTPUT_DIR
    path = pathlib.Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def default_jobs_path(cfg: Dict) -> pathlib.Path:
    return get_output_dir(cfg) / "metadata" / "jobs.jsonl"


def is_under(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def generate_job_id(idx: int, seed: int) -> str:
    h = hashlib.sha256("{}_{}".format(seed, idx).encode()).hexdigest()[:12]
    return "job_{:06d}_{}".format(idx, h)


def create_job(idx: int, cfg: Dict, output_root: pathlib.Path = None) -> Dict:
    seed = cfg["dataset"]["seed"]
    rng = random.Random(seed + idx)
    cats = cfg["targets"]["categories"]
    if cfg.get("targets", {}).get("selection_mode") == "sequential":
        cat = cats[idx % len(cats)]
    else:
        cat = rng.choice(cats)
    lo, hi = cat["scale_range_m"]
    n_cam = rng.randint(cfg["dataset"]["num_cameras_min"], cfg["dataset"]["num_cameras_max"])
    occ = cfg.get("occluders", {})
    occ_range = occ.get("count_range", [2, 8])
    n_occ = rng.randint(*occ_range)
    occ_cats = [rng.choice(occ.get("categories", [{"name": "wall"}]))["name"] for _ in range(n_occ)]
    shard = idx // cfg["output"]["shard_size_sequences"]
    output_root = output_root or get_output_dir(cfg)
    out = str(output_root / "shards" / "shard_{:06d}".format(shard) / "seq_{:06d}".format(idx))
    return {
        "job_id": generate_job_id(idx, seed), "sequence_id": "seq_{:06d}".format(idx),
        "sequence_index": idx, "seed": seed + idx,
        "target_category": cat["name"], "target_mesh": cat["mesh"],
        "target_ground_z_cm": float(cat.get("ground_z_cm", 0.0)),
        "target_scale_m": round(rng.uniform(lo, hi), 3),
        "target_motion_type": cat["motion_type"],
        "num_cameras": n_cam, "num_frames": cfg["dataset"]["frames_per_sequence"],
        "fps": cfg["dataset"]["fps"],
        "resolution": [cfg["dataset"]["resolution"]["width"], cfg["dataset"]["resolution"]["height"]],
        "num_occluders": n_occ, "occluder_categories": occ_cats,
        "output_dir": out, "status": "pending", "retry_count": 0, "max_retries": 3,
    }


def save_jobs(jobs: List[Dict], path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


def load_jobs(path: str) -> List[Dict]:
    jobs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def update_job(jp: str, jid: str, status: str, error: str = None):
    jobs = load_jobs(jp)
    for j in jobs:
        if j["job_id"] == jid:
            j["status"] = status
            if error:
                j["last_error"] = error
            if status == "failed":
                j["retry_count"] = j.get("retry_count", 0) + 1
            break
    save_jobs(jobs, jp)


def run_worker(job: Dict, timeout: int = 600) -> int:
    jp = pathlib.Path("/tmp/mvtrack_jobs") / "{}.json".format(job["job_id"])
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(job, indent=2))
    pathlib.Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
    lp = pathlib.Path(job["output_dir"]) / "ue.log"
    print("[Orchestrator] {} -> {}".format(job["job_id"], job["output_dir"]))
    try:
        r = subprocess.run(
            ["bash", WRAPPER, str(jp), str(lp), str(timeout)],
            timeout=timeout + 60, capture_output=True, text=True
        )
        lines = r.stdout.strip().split("\n")
        try:
            return int(lines[-1])
        except Exception:
            return r.returncode
    except subprocess.TimeoutExpired:
        return -1
    except Exception as e:
        print("  ERROR: {}".format(e))
        return -2


def validate(job: Dict) -> Tuple[bool, str]:
    out = pathlib.Path(job["output_dir"])
    mp = out / "seq_meta.json"
    if not mp.exists():
        return False, "MISSING_SEQ_META"
    try:
        m = json.loads(mp.read_text())
    except Exception:
        return False, "CORRUPT_SEQ_META"
    if not m.get("success", False):
        return False, "UE_REPORTED_FAILURE"
    expected_frames = job.get("num_frames", 0)
    rendered = int(m.get("rendered_rgb_frames", 0))
    if rendered < expected_frames:
        return False, "FEW_RENDERED_RGB_FRAMES:{}/{}".format(rendered, expected_frames)
    cd = out / "cameras"
    if not cd.exists():
        return False, "MISSING_CAMERAS"
    cfs = list(cd.glob("cam_*.json"))
    if len(cfs) < job["num_cameras"]:
        return False, "MISSING_CALIBRATIONS"
    frames = out / "frames"
    for cam_idx in range(job["num_cameras"]):
        rgb_dir = frames / "cam_{:03d}".format(cam_idx) / "rgb"
        rgb_count = len(list(rgb_dir.glob("*.png"))) if rgb_dir.exists() else 0
        if rgb_count < expected_frames:
            return False, "FEW_RGB_FILES:cam_{:03d}:{}/{}".format(cam_idx, rgb_count, expected_frames)
    return True, "OK"


def run_generation(cfg: Dict, dry_run: bool = False, force_regenerate: bool = False, jobs_path: str = None):
    output_root = get_output_dir(cfg)
    jp_path = pathlib.Path(jobs_path).expanduser() if jobs_path else default_jobs_path(cfg)
    if not jp_path.is_absolute():
        jp_path = ROOT / jp_path
    jp = str(jp_path.resolve())

    regenerate = force_regenerate or not jp_path.exists()
    if jp_path.exists() and not force_regenerate:
        jobs = load_jobs(jp)
        expected_n = cfg["dataset"]["num_sequences"]
        mismatched_outputs = [
            j.get("output_dir", "") for j in jobs
            if not is_under(pathlib.Path(j.get("output_dir", "")), output_root)
        ]
        if len(jobs) != expected_n:
            print("[Orchestrator] Existing manifest has {} jobs but config requests {}; regenerating".format(
                len(jobs), expected_n))
            regenerate = True
        elif mismatched_outputs:
            print("[Orchestrator] Existing manifest output_dir does not match {}; regenerating".format(output_root))
            regenerate = True
    else:
        jobs = []

    if regenerate:
        jobs = [create_job(i, cfg, output_root=output_root) for i in range(cfg["dataset"]["num_sequences"])]
        save_jobs(jobs, jp)
        print("[Orchestrator] Generated {} jobs".format(len(jobs)))
        print("[Orchestrator] Output root: {}".format(output_root))

    pending = [j for j in jobs if j["status"] in ("pending", "failed")
               and j.get("retry_count", 0) < j.get("max_retries", 3)]
    print("[Orchestrator] {} jobs to process".format(len(pending)))

    if dry_run:
        print("[Orchestrator] DRY RUN")
        for j in pending[:5]:
            print("  {}: {} scale={}".format(j["job_id"], j["target_category"], j["target_scale_m"]))
        return

    ok_count = 0
    fail_count = 0
    for job in pending:
        print("\n" + "=" * 60)
        print("[Orchestrator] {} ({}/{})".format(job["job_id"], ok_count + fail_count + 1, len(pending)))
        print("  Target: {} scale={}m".format(job["target_category"], job["target_scale_m"]))

        update_job(jp, job["job_id"], "running")
        t0 = time.time()
        rc = run_worker(job)
        elapsed = time.time() - t0

        if rc != 0:
            print("  FAILED (exit {}, {:.1f}s)".format(rc, elapsed))
            update_job(jp, job["job_id"], "failed", "exit={}".format(rc))
            fail_count += 1
            continue

        vok, vreason = validate(job)
        if not vok:
            print("  QC FAILED: {}".format(vreason))
            update_job(jp, job["job_id"], "failed", "QC:{}".format(vreason))
            fail_count += 1
            continue

        print("  OK ({:.1f}s)".format(elapsed))
        update_job(jp, job["job_id"], "done")
        ok_count += 1

    print("\n" + "=" * 60)
    print("[Orchestrator] COMPLETE: {} success, {} failed".format(ok_count, fail_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIGS_DIR / "dataset.yaml"))
    parser.add_argument("--mode", choices=["generate", "report"], default="generate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Rebuild jobs.jsonl from the current config, ignoring any stale manifest.")
    parser.add_argument("--jobs-path", default=None,
                        help="Optional explicit jobs manifest path. Defaults to <output.root>/metadata/jobs.jsonl.")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.mode == "generate":
        run_generation(cfg, dry_run=args.dry_run,
                       force_regenerate=args.force_regenerate,
                       jobs_path=args.jobs_path)
    elif args.mode == "report":
        jp = str((pathlib.Path(args.jobs_path).expanduser() if args.jobs_path else default_jobs_path(cfg)).resolve())
        if pathlib.Path(jp).exists():
            jobs = load_jobs(jp)
            bs = {}
            for j in jobs:
                bs[j["status"]] = bs.get(j["status"], 0) + 1
            print("\n[Dataset Report]")
            print("  Total: {}".format(len(jobs)))
            for s, c in sorted(bs.items()):
                print("  {}: {}".format(s, c))


if __name__ == "__main__":
    main()
