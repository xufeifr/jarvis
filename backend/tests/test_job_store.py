"""Tests for deerflow.factory.job_store — YAML CRUD operations."""

import pytest

from deerflow.factory.job_store import delete_job, list_jobs, load_job, save_job
from deerflow.factory.models import JobDescription


def _make_job(job_id: str = "test-job") -> JobDescription:
    return JobDescription(
        job_id=job_id,
        job_name="测试岗位",
        responsibilities=["处理请求"],
    )


@pytest.fixture(autouse=True)
def _use_tmp_paths(tmp_path, monkeypatch):
    """Redirect all factory storage to a temp directory."""
    from deerflow.config import paths

    monkeypatch.setattr(paths, "_paths", paths.Paths(base_dir=tmp_path))
    yield


class TestSaveAndLoad:
    def test_save_creates_file(self, tmp_path):
        job = _make_job()
        path = save_job(job)
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_load_returns_same_data(self):
        job = _make_job()
        save_job(job)
        loaded = load_job("test-job")
        assert loaded.job_id == job.job_id
        assert loaded.job_name == job.job_name
        assert loaded.responsibilities == job.responsibilities

    def test_load_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Job not found"):
            load_job("nonexistent")


class TestListJobs:
    def test_empty_list(self):
        assert list_jobs() == []

    def test_multiple_jobs(self):
        save_job(_make_job("job-a"))
        save_job(_make_job("job-b"))
        jobs = list_jobs()
        ids = [j.job_id for j in jobs]
        assert "job-a" in ids
        assert "job-b" in ids


class TestDeleteJob:
    def test_delete_existing(self):
        save_job(_make_job())
        assert delete_job("test-job") is True
        with pytest.raises(FileNotFoundError):
            load_job("test-job")

    def test_delete_nonexistent(self):
        assert delete_job("nonexistent") is False
