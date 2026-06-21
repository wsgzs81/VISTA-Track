#!/usr/bin/env python3
"""Run multiple MVTrack UE workers over one jobs manifest.

This is for generation throughput after a config has passed visual QC. It keeps
the existing JSONL job format and uses simple file locking to avoid duplicate
work across processes.
"""
import argparse
import json
import pathlib
import subprocess
import time
from typing import Dict, List

import yaml

from orchestrator import default_jobs_path, get_output_dir, load_jobs, run_worker, save_jobs, validate, create_job


ROOT = pathlib.Path(__file__).resolve().parents[2]


def locked(lock_path: pathlib.Path):
    import fcntl

    class Lock:
        def __enter__(self):
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(lock_path, "w")
            fcntl.flock(self.f, fcntl.LOCK_EX)
            return self

        def __exit__(self, exc_type, exc, tb):
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()

    return Lock()


def ensure_jobs(cfg: Dict, jobs_path: pathlib.Path, force_regenerate: bool) -> None:
    if jobs_path.exists() and not force_regenerate:
        return
    output_root = get_output_dir(cfg)
    jobs = [create_job(i, cfg, output_root=output_root) for i in range(cfg["dataset"]["num_sequences"])]
    save_jobs(jobs, str(jobs_path))


def claim_job(jobs_path: pathlib.Path, worker_id: int) -> Dict:
    jobs = load_jobs(str(jobs_path))
    for job in jobs:
        if job["status"] in ("pending", "failed") and job.get("retry_count", 0) < job.get("max_retries", 3):
            job["status"] = "running"
            job["worker_id"] = worker_id
            job["started_at"] = time.time()
            save_jobs(jobs, str(jobs_path))
            return job
    return {}


def finish_job(jobs_path: pathlib.Path, job_id: str, status: str, error: str = "") -> None:
    jobs = load_jobs(str(jobs_path))
    for job in jobs:
        if job["job_id"] == job_id:
            job["status"] = status
            job["finished_at"] = time.time()
            if error:
                job["last_error"] = error
            if status == "failed":
                job["retry_count"] = job.get("retry_count", 0) + 1
            break
    save_jobs(jobs, str(jobs_path))


def worker_loop(jobs_path: pathlib.Path, lock_path: pathlib.Path, worker_id: int, timeout: int) -> None:
    while True:
        with locked(lock_path):
            job = claim_job(jobs_path, worker_id)
        if not job:
            print(f"[worker {worker_id}] no more jobs")
            return

        print(f"[worker {worker_id}] start {job['sequence_id']} {job['target_category']}")
        rc = run_worker(job, timeout=timeout)
        if rc != 0:
            with locked(lock_path):
                finish_job(jobs_path, job["job_id"], "failed", f"exit={rc}")
            print(f"[worker {worker_id}] failed {job['sequence_id']} exit={rc}")
            continue

        ok, reason = validate(job)
        with locked(lock_path):
            finish_job(jobs_path, job["job_id"], "done" if ok else "failed", "" if ok else f"QC:{reason}")
        print(f"[worker {worker_id}] {'done' if ok else 'qc-failed'} {job['sequence_id']} {reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs-path", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--worker-id", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    jobs_path = pathlib.Path(args.jobs_path).expanduser() if args.jobs_path else default_jobs_path(cfg)
    if not jobs_path.is_absolute():
        jobs_path = ROOT / jobs_path
    lock_path = jobs_path.with_suffix(jobs_path.suffix + ".lock")

    ensure_jobs(cfg, jobs_path, args.force_regenerate)

    if args.worker_id is not None:
        worker_loop(jobs_path, lock_path, args.worker_id, args.timeout)
        return

    procs: List[subprocess.Popen] = []
    for worker_id in range(args.workers):
        cmd = [
            "python3", str(pathlib.Path(__file__).resolve()),
            "--config", args.config,
            "--jobs-path", str(jobs_path),
            "--timeout", str(args.timeout),
            "--worker-id", str(worker_id),
        ]
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
    rc = 0
    for proc in procs:
        rc = max(rc, proc.wait())
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
