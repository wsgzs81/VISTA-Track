"""Job manifest generation and management.

Creates job manifests for each sequence, handles sharding,
and manages the job lifecycle (pending -> running -> done/failed).
"""
import json
import hashlib
import pathlib
import random
from typing import List, Dict, Any, Optional


def generate_job_id(sequence_index: int, seed: int) -> str:
    """Deterministic job ID from sequence index and seed."""
    h = hashlib.sha256(f"{seed}_{sequence_index}".encode()).hexdigest()[:12]
    return f"job_{sequence_index:06d}_{h}"


def create_job_manifest(
    sequence_index: int,
    dataset_cfg: Dict[str, Any],
    cameras_cfg: Dict[str, Any],
    render_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a single job manifest for one sequence."""
    seed = dataset_cfg["dataset"]["seed"]
    rng = random.Random(seed + sequence_index)

    # Select target category
    categories = dataset_cfg["targets"]["categories"]
    target_cat = rng.choice(categories)

    # Randomize target scale within category range
    scale_lo, scale_hi = target_cat["scale_range_m"]
    target_scale_m = round(rng.uniform(scale_lo, scale_hi), 3)

    # Camera count
    n_cam_min = dataset_cfg["dataset"]["num_cameras_min"]
    n_cam_max = dataset_cfg["dataset"]["num_cameras_max"]
    num_cameras = rng.randint(n_cam_min, n_cam_max)

    # Occluder count
    occ_cfg = dataset_cfg.get("occluders", {})
    occ_count_range = occ_cfg.get("count_range", [2, 8])
    num_occluders = rng.randint(*occ_count_range)

    # Output directory
    shard_index = sequence_index // dataset_cfg["output"]["shard_size_sequences"]
    shard_dir = f"shard_{shard_index:06d}"
    seq_dir = f"seq_{sequence_index:06d}"
    output_root = dataset_cfg["output"]["root"]
    output_dir = str(pathlib.Path(output_root) / "shards" / shard_dir / seq_dir)

    job = {
        "job_id": generate_job_id(sequence_index, seed),
        "sequence_id": f"seq_{sequence_index:06d}",
        "sequence_index": sequence_index,
        "seed": seed + sequence_index,
        "target_category": target_cat["name"],
        "target_mesh": target_cat["mesh"],
        "target_scale_m": target_scale_m,
        "target_motion_type": target_cat["motion_type"],
        "num_cameras": num_cameras,
        "num_frames": dataset_cfg["dataset"]["frames_per_sequence"],
        "fps": dataset_cfg["dataset"]["fps"],
        "resolution": [
            dataset_cfg["dataset"]["resolution"]["width"],
            dataset_cfg["dataset"]["resolution"]["height"],
        ],
        "num_occluders": num_occluders,
        "occluder_categories": [
            rng.choice(occ_cfg.get("categories", [{"name": "wall_segment"}]))["name"]
            for _ in range(num_occluders)
        ],
        "output_dir": output_dir,
        "status": "pending",
        "retry_count": 0,
        "max_retries": 3,
    }
    return job


def generate_all_jobs(
    dataset_cfg: Dict[str, Any],
    cameras_cfg: Dict[str, Any],
    render_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate all job manifests for the dataset."""
    num_sequences = dataset_cfg["dataset"]["num_sequences"]
    jobs = []
    for i in range(num_sequences):
        job = create_job_manifest(i, dataset_cfg, cameras_cfg, render_cfg)
        jobs.append(job)
    return jobs


def save_jobs(jobs: List[Dict[str, Any]], path: str) -> None:
    """Save job list to JSONL file."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")


def load_jobs(path: str) -> List[Dict[str, Any]]:
    """Load job list from JSONL file."""
    jobs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def update_job_status(jobs_path: str, job_id: str, status: str,
                      error: Optional[str] = None) -> None:
    """Update a single job's status in the JSONL file."""
    jobs = load_jobs(jobs_path)
    for job in jobs:
        if job["job_id"] == job_id:
            job["status"] = status
            if error:
                job["last_error"] = error
            if status == "failed":
                job["retry_count"] = job.get("retry_count", 0) + 1
            break
    save_jobs(jobs, jobs_path)


if __name__ == "__main__":
    import yaml
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dataset.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cameras_cfg = {}
    render_cfg = {}
    jobs = generate_all_jobs(cfg, cameras_cfg, render_cfg)
    out_path = str(pathlib.Path(cfg["output"]["root"]) / "metadata" / "jobs.jsonl")
    save_jobs(jobs, out_path)
    print(f"Generated {len(jobs)} jobs -> {out_path}")
