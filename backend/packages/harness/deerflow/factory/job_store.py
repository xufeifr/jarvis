"""YAML-based storage for job descriptions.

Stores job descriptions as YAML files under `.deer-flow/factory/jobs/`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from deerflow.config.paths import get_paths
from deerflow.factory.models import JobDescription

logger = logging.getLogger(__name__)


def _jobs_dir() -> Path:
    """Return the jobs storage directory, creating it if needed."""
    d = get_paths().base_dir / "factory" / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.yaml"


def save_job(job: JobDescription) -> Path:
    """Persist a job description as a YAML file.

    Returns:
        The path to the written file.
    """
    path = _job_path(job.job_id)
    data = job.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    logger.info("Saved job %s → %s", job.job_id, path)
    return path


def load_job(job_id: str) -> JobDescription:
    """Load a job description from disk.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    path = _job_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"Job not found: {job_id} (looked at {path})")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return JobDescription(**data)


def list_jobs() -> list[JobDescription]:
    """Return all stored job descriptions."""
    jobs_dir = _jobs_dir()
    results: list[JobDescription] = []
    for entry in sorted(jobs_dir.glob("*.yaml")):
        try:
            with open(entry, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            results.append(JobDescription(**data))
        except Exception as e:
            logger.warning("Skipping invalid job file %s: %s", entry.name, e)
    return results


def delete_job(job_id: str) -> bool:
    """Delete a job description file.

    Returns:
        True if the file existed and was deleted, False otherwise.
    """
    path = _job_path(job_id)
    if path.exists():
        path.unlink()
        logger.info("Deleted job %s", job_id)
        return True
    return False
