"""Tests for deerflow.factory.factory — end-to-end factory pipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from deerflow.factory.factory import DigitalWorkerFactory
from deerflow.factory.models import (
    AssemblyResult,
    CapabilityMatrix,
    ExamResult,
    ExamScores,
    JobDescription,
)


@pytest.fixture(autouse=True)
def _use_tmp_paths(tmp_path, monkeypatch):
    from deerflow.config import paths

    monkeypatch.setattr(paths, "_paths", paths.Paths(base_dir=tmp_path))
    yield


def _make_job() -> JobDescription:
    return JobDescription(
        job_id="test-handler",
        job_name="测试处理专员",
        responsibilities=["处理测试请求"],
    )


class TestDigitalWorkerFactory:
    @pytest.mark.anyio
    async def test_create_worker_passing(self):
        factory = DigitalWorkerFactory()

        # Mock analyzer
        mock_matrix = CapabilityMatrix(job_id="test-handler")
        factory._analyzer = MagicMock()
        factory._analyzer.analyze.return_value = mock_matrix

        # Mock assembler
        factory._assembler = MagicMock()
        factory._assembler.assemble.return_value = AssemblyResult(
            worker_id="w-test-001",
            agent_name="factory-test-handler",
        )

        # Mock exam runner
        factory._exam_runner = MagicMock()
        factory._exam_runner.run_exam = AsyncMock(
            return_value=ExamResult(
                exam_id="e-001",
                worker_id="w-test-001",
                job_id="test-handler",
                passed=True,
                scores=ExamScores(overall=85.0),
            )
        )

        result = await factory.create_worker(_make_job())

        assert result.level == "L3"
        assert result.worker_id == "w-test-001"
        factory._analyzer.analyze.assert_called_once()
        factory._assembler.assemble.assert_called_once()

    @pytest.mark.anyio
    async def test_create_worker_failing_all_reworks(self):
        factory = DigitalWorkerFactory()

        mock_matrix = CapabilityMatrix(job_id="test-handler")
        factory._analyzer = MagicMock()
        factory._analyzer.analyze.return_value = mock_matrix

        factory._assembler = MagicMock()
        factory._assembler.assemble.return_value = AssemblyResult(
            worker_id="w-test-002",
            agent_name="factory-test-handler",
        )

        # Always fail exam
        factory._exam_runner = MagicMock()
        factory._exam_runner.run_exam = AsyncMock(
            return_value=ExamResult(
                exam_id="e-fail",
                worker_id="w-test-002",
                job_id="test-handler",
                passed=False,
                scores=ExamScores(overall=30.0),
                diagnosis="Poor performance",
            )
        )

        result = await factory.create_worker(_make_job())

        # After 3 rework attempts, still at L2
        assert result.level == "L2"
        assert result.rework_count == 3

    def test_analyze_job(self):
        factory = DigitalWorkerFactory()
        factory._analyzer = MagicMock()
        factory._analyzer.analyze.return_value = CapabilityMatrix(job_id="test")

        result = factory.analyze_job(_make_job())
        assert result.job_id == "test"

    def test_assemble_agent(self):
        factory = DigitalWorkerFactory()
        factory._assembler = MagicMock()
        factory._assembler.assemble.return_value = AssemblyResult(worker_id="w-001", agent_name="factory-test")

        result = factory.assemble_agent(_make_job())
        assert result.worker_id == "w-001"
