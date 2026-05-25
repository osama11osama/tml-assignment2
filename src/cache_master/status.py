"""Resumable progress tracking and multi-worker coordination."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.cache_master.paths import (
    ALL_STAGES,
    CLAIMS_DIR,
    PROGRESS_PATH,
    STAGE_FILES,
    claim_path,
    manifest_path,
    stage_path,
    suspect_dir,
)
from src.paths import NUM_SUSPECTS

CLAIM_STALE_SEC = 7200  # 2 h — reclaim if worker died


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(root: Path) -> dict:
    mp = manifest_path(root)
    if not mp.exists():
        return {"stages": {}, "complete": False}
    return json.loads(mp.read_text(encoding="utf-8"))


def save_manifest(root: Path, manifest: dict) -> None:
    mp = manifest_path(root)
    mp.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(mp)


def stage_done(root: Path, stage: str) -> bool:
    return stage_path(root, stage).exists()


def is_suspect_complete(suspect_id: int) -> bool:
    root = suspect_dir(suspect_id)
    mf = load_manifest(root)
    if mf.get("complete"):
        return True
    return all(stage_done(root, s) for s in ALL_STAGES)


def mark_stage(root: Path, stage: str, worker: str) -> None:
    mf = load_manifest(root)
    mf.setdefault("stages", {})[stage] = {"at": _utc_now(), "worker": worker}
    mf["complete"] = all(stage in mf["stages"] for stage in ALL_STAGES)
    if mf["complete"]:
        mf["completed_at"] = _utc_now()
    save_manifest(root, mf)


def suspects_for_worker(worker_index: int, num_workers: int) -> list[int]:
    return [i for i in range(NUM_SUSPECTS) if i % num_workers == worker_index]


def try_claim(suspect_id: int, worker_name: str) -> bool:
    """File lock for parallel workers (same or different machines)."""
    if is_suspect_complete(suspect_id):
        return False
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    cp = claim_path(suspect_id)
    if cp.exists():
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            age = time.time() - data.get("ts", 0)
            if age < CLAIM_STALE_SEC and data.get("worker") != worker_name:
                return False
        except (json.JSONDecodeError, OSError):
            pass
    cp.write_text(
        json.dumps({"worker": worker_name, "ts": time.time(), "at": _utc_now()}),
        encoding="utf-8",
    )
    return True


def release_claim(suspect_id: int, worker_name: str) -> None:
    cp = claim_path(suspect_id)
    if cp.exists():
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            if data.get("worker") == worker_name:
                cp.unlink(missing_ok=True)
        except (json.JSONDecodeError, OSError):
            cp.unlink(missing_ok=True)


def count_complete() -> tuple[int, int]:
    done = sum(1 for i in range(NUM_SUSPECTS) if is_suspect_complete(i))
    return done, NUM_SUSPECTS


def list_incomplete(worker_index: int | None = None, num_workers: int | None = None) -> list[int]:
    ids = range(NUM_SUSPECTS)
    if worker_index is not None and num_workers is not None:
        ids = suspects_for_worker(worker_index, num_workers)
    return [i for i in ids if not is_suspect_complete(i)]


def stage_summary(suspect_id: int) -> dict[str, bool]:
    root = suspect_dir(suspect_id)
    return {s: stage_done(root, s) for s in ALL_STAGES}


def write_progress(worker_name: str, last_suspect: int | None = None) -> None:
    done, total = count_complete()
    payload = {
        "updated_at": _utc_now(),
        "worker": worker_name,
        "complete": done,
        "total": total,
        "last_suspect": last_suspect,
        "pct": round(100.0 * done / total, 2),
    }
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_status(worker_index: int | None = None, num_workers: int | None = None) -> None:
    done, total = count_complete()
    print(f"Master cache: {done}/{total} suspects complete ({100*done/total:.1f}%)")
    if PROGRESS_PATH.exists():
        prog = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        print(f"  Last update: {prog.get('updated_at')} by {prog.get('worker')}")
    pending = list_incomplete(worker_index, num_workers)
    if worker_index is not None:
        print(f"  This worker ({worker_index}/{num_workers}): {len(pending)} pending")
    if pending[:5]:
        print(f"  Next pending ids: {pending[:10]}{'...' if len(pending) > 10 else ''}")
