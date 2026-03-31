"""Tests for deerflow.factory.onboarding — L0-L6 state machine."""

import pytest

from deerflow.factory.models import CaseRecord, WorkerStats, WorkerStatus
from deerflow.factory.onboarding import OnboardingManager


@pytest.fixture(autouse=True)
def _use_tmp_paths(tmp_path, monkeypatch):
    from deerflow.config import paths

    monkeypatch.setattr(paths, "_paths", paths.Paths(base_dir=tmp_path))
    yield


def _make_worker(level="L0", **overrides) -> WorkerStatus:
    defaults = {
        "worker_id": "w-001",
        "job_id": "test",
        "agent_name": "factory-test",
        "level": level,
    }
    defaults.update(overrides)
    return WorkerStatus(**defaults)


class TestSaveAndGet:
    def test_save_and_get(self):
        mgr = OnboardingManager()
        worker = _make_worker()
        mgr.save_worker(worker)
        loaded = mgr.get_worker("w-001")
        assert loaded.worker_id == "w-001"
        assert loaded.level == "L0"

    def test_get_not_found(self):
        mgr = OnboardingManager()
        with pytest.raises(FileNotFoundError, match="Worker not found"):
            mgr.get_worker("nonexistent")


class TestListWorkers:
    def test_empty(self):
        mgr = OnboardingManager()
        assert mgr.list_workers() == []

    def test_multiple(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(worker_id="w-001"))
        mgr.save_worker(_make_worker(worker_id="w-002"))
        workers = mgr.list_workers()
        assert len(workers) == 2


class TestPromote:
    def test_l0_to_l1_auto(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(level="L0"))
        result = mgr.promote("w-001")
        assert result.level == "L1"

    def test_l1_to_l2_auto(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(level="L1"))
        result = mgr.promote("w-001")
        assert result.level == "L2"

    def test_l2_to_l3_auto(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(level="L2"))
        result = mgr.promote("w-001")
        assert result.level == "L3"

    def test_l3_to_l4_requires_criteria(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(level="L3"))
        with pytest.raises(ValueError, match="does not meet promotion criteria"):
            mgr.promote("w-001")

    def test_l3_to_l4_with_criteria_met(self):
        mgr = OnboardingManager()
        worker = _make_worker(
            level="L3",
            stats=WorkerStats(consecutive_success=100, total_cases=100, successful_cases=100),
        )
        mgr.save_worker(worker)
        result = mgr.promote("w-001")
        assert result.level == "L4"

    def test_l6_cannot_promote(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(level="L6"))
        with pytest.raises(ValueError, match="maximum level"):
            mgr.promote("w-001")


class TestCheckPromotion:
    def test_auto_levels_return_true(self):
        mgr = OnboardingManager()
        for level in ("L0", "L1", "L2"):
            mgr.save_worker(_make_worker(level=level))
            assert mgr.check_promotion("w-001") is True

    def test_l3_without_criteria(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(level="L3"))
        assert mgr.check_promotion("w-001") is False

    def test_l6_returns_false(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(level="L6"))
        assert mgr.check_promotion("w-001") is False


class TestRecordCase:
    def test_success_increments(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(level="L3"))
        record = CaseRecord(case_id="c-001", success=True)
        worker = mgr.record_case("w-001", record)
        assert worker.stats.total_cases == 1
        assert worker.stats.successful_cases == 1
        assert worker.stats.consecutive_success == 1

    def test_failure_resets_consecutive(self):
        mgr = OnboardingManager()
        w = _make_worker(level="L3", stats=WorkerStats(consecutive_success=50))
        mgr.save_worker(w)
        record = CaseRecord(case_id="c-002", success=False)
        worker = mgr.record_case("w-001", record)
        assert worker.stats.consecutive_success == 0
        assert worker.stats.failed_cases == 1

    def test_new_scenario_counted(self):
        mgr = OnboardingManager()
        mgr.save_worker(_make_worker(level="L3"))
        record = CaseRecord(case_id="c-003", success=True, is_new_scenario=True)
        worker = mgr.record_case("w-001", record)
        assert worker.stats.new_scenario_count == 1
