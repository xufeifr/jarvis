"""Onboarding Manager — L0-L6 state machine for digital workers.

Manages the lifecycle of a digital worker from assembly (L0) through
production readiness (L6), including promotion checks, case recording,
and periodic health checks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from deerflow.config.paths import get_paths
from deerflow.factory.models import CaseRecord, WorkerLevel, WorkerStatus

logger = logging.getLogger(__name__)

# ── Promotion thresholds ─────────────────────────────────────────────────

PROMOTION_RULES: dict[WorkerLevel, dict] = {
    "L0": {"next": "L1", "auto": True, "description": "Assembly complete"},
    "L1": {"next": "L2", "auto": True, "description": "Exam started"},
    "L2": {"next": "L3", "auto": True, "description": "Exam passed"},
    "L3": {
        "next": "L4",
        "auto": False,
        "description": "100 consecutive cases without major error",
        "consecutive_success": 100,
    },
    "L4": {
        "next": "L5",
        "auto": False,
        "description": "1000+ total cases, edge-case accuracy >= 85%",
        "total_cases": 1000,
        "edge_case_accuracy": 0.85,
    },
    "L5": {
        "next": "L6",
        "auto": False,
        "description": "5000+ total cases, 10+ new scenario types",
        "total_cases": 5000,
        "new_scenario_count": 10,
    },
}


def _workers_dir() -> Path:
    d = get_paths().base_dir / "factory" / "workers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _worker_path(worker_id: str) -> Path:
    return _workers_dir() / f"{worker_id}.json"


class OnboardingManager:
    """Manages digital worker lifecycle and level progression."""

    def save_worker(self, worker: WorkerStatus) -> Path:
        """Persist worker status to disk."""
        path = _worker_path(worker.worker_id)
        path.write_text(worker.model_dump_json(indent=2), encoding="utf-8")
        return path

    def get_worker(self, worker_id: str) -> WorkerStatus:
        """Load a worker's status from disk.

        Raises:
            FileNotFoundError: If the worker does not exist.
        """
        path = _worker_path(worker_id)
        if not path.exists():
            raise FileNotFoundError(f"Worker not found: {worker_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return WorkerStatus(**data)

    def list_workers(self) -> list[WorkerStatus]:
        """Return all stored workers."""
        results: list[WorkerStatus] = []
        for entry in sorted(_workers_dir().glob("*.json")):
            try:
                data = json.loads(entry.read_text(encoding="utf-8"))
                results.append(WorkerStatus(**data))
            except Exception as e:
                logger.warning("Skipping invalid worker file %s: %s", entry.name, e)
        return results

    def promote(self, worker_id: str) -> WorkerStatus:
        """Promote a worker to the next level.

        Raises:
            ValueError: If the worker is already at the max level or
                        does not meet promotion criteria.
        """
        worker = self.get_worker(worker_id)

        if worker.level == "L6":
            raise ValueError(f"Worker {worker_id} is already at maximum level L6")

        rule = PROMOTION_RULES.get(worker.level)
        if rule is None:
            raise ValueError(f"No promotion rule for level {worker.level}")

        if not rule.get("auto", False) and not self._check_criteria(worker, rule):
            raise ValueError(f"Worker {worker_id} does not meet promotion criteria for {worker.level} → {rule['next']}: {rule['description']}")

        worker.level = rule["next"]
        self.save_worker(worker)
        logger.info("Promoted worker %s to %s", worker_id, worker.level)
        return worker

    def check_promotion(self, worker_id: str) -> bool:
        """Check if a worker meets the criteria for next-level promotion."""
        worker = self.get_worker(worker_id)
        if worker.level == "L6":
            return False
        rule = PROMOTION_RULES.get(worker.level)
        if rule is None:
            return False
        if rule.get("auto", False):
            return True
        return self._check_criteria(worker, rule)

    def record_case(self, worker_id: str, record: CaseRecord) -> WorkerStatus:
        """Record a production case result and update stats."""
        worker = self.get_worker(worker_id)
        stats = worker.stats

        stats.total_cases += 1
        if record.success:
            stats.successful_cases += 1
            stats.consecutive_success += 1
        else:
            stats.failed_cases += 1
            stats.consecutive_success = 0

        if record.is_new_scenario:
            stats.new_scenario_count += 1

        # Recalculate accuracy
        if stats.total_cases > 0:
            stats.accuracy_rate = round(stats.successful_cases / stats.total_cases, 4)

        self.save_worker(worker)
        return worker

    def _check_criteria(self, worker: WorkerStatus, rule: dict) -> bool:
        """Check whether a worker meets specific promotion criteria."""
        stats = worker.stats

        if "consecutive_success" in rule:
            if stats.consecutive_success < rule["consecutive_success"]:
                return False

        if "total_cases" in rule:
            if stats.total_cases < rule["total_cases"]:
                return False

        if "edge_case_accuracy" in rule:
            if stats.edge_case_accuracy < rule["edge_case_accuracy"]:
                return False

        if "new_scenario_count" in rule:
            if stats.new_scenario_count < rule["new_scenario_count"]:
                return False

        return True
