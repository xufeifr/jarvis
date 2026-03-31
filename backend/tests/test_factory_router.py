"""Tests for app.gateway.routers.factory — API endpoint tests."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _use_tmp_paths(tmp_path, monkeypatch):
    from deerflow.config import paths

    monkeypatch.setattr(paths, "_paths", paths.Paths(base_dir=tmp_path))
    yield


@pytest.fixture
def client():
    """Create a test client for the factory router."""
    from fastapi import FastAPI

    from app.gateway.routers.factory import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _job_payload(job_id="test-job"):
    return {
        "job_id": job_id,
        "job_name": "测试岗位",
        "responsibilities": ["处理请求"],
    }


class TestJobEndpoints:
    def test_list_jobs_empty(self, client):
        resp = client.get("/api/factory/jobs")
        assert resp.status_code == 200
        assert resp.json()["jobs"] == []

    def test_create_and_get_job(self, client):
        resp = client.post("/api/factory/jobs", json=_job_payload())
        assert resp.status_code == 201
        assert resp.json()["job_id"] == "test-job"

        resp = client.get("/api/factory/jobs/test-job")
        assert resp.status_code == 200
        assert resp.json()["job_name"] == "测试岗位"

    def test_list_jobs_after_create(self, client):
        client.post("/api/factory/jobs", json=_job_payload("job-a"))
        client.post("/api/factory/jobs", json=_job_payload("job-b"))
        resp = client.get("/api/factory/jobs")
        assert resp.status_code == 200
        assert len(resp.json()["jobs"]) == 2

    def test_get_job_not_found(self, client):
        resp = client.get("/api/factory/jobs/nonexistent")
        assert resp.status_code == 404

    def test_delete_job(self, client):
        client.post("/api/factory/jobs", json=_job_payload())
        resp = client.delete("/api/factory/jobs/test-job")
        assert resp.status_code == 204

    def test_delete_job_not_found(self, client):
        resp = client.delete("/api/factory/jobs/nonexistent")
        assert resp.status_code == 404


class TestWorkerEndpoints:
    def test_list_workers_empty(self, client):
        resp = client.get("/api/factory/workers")
        assert resp.status_code == 200
        assert resp.json()["workers"] == []

    def test_get_worker_not_found(self, client):
        resp = client.get("/api/factory/workers/nonexistent")
        assert resp.status_code == 404
