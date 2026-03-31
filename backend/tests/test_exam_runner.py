"""Tests for deerflow.factory.exam_runner — exam orchestration."""

from unittest.mock import AsyncMock, patch

import pytest

from deerflow.factory.exam_runner import ExamRunner, _save_exam_result, load_exam_result
from deerflow.factory.models import (
    CaseResult,
    ExamCase,
    ExamPlan,
    ExamResult,
    ExamScores,
    JobDescription,
    Message,
    WorkerStatus,
)


@pytest.fixture(autouse=True)
def _use_tmp_paths(tmp_path, monkeypatch):
    from deerflow.config import paths

    monkeypatch.setattr(paths, "_paths", paths.Paths(base_dir=tmp_path))
    yield


def _make_worker() -> WorkerStatus:
    return WorkerStatus(
        worker_id="w-001",
        job_id="test-handler",
        agent_name="factory-test-handler",
        level="L2",
    )


def _make_job() -> JobDescription:
    return JobDescription(
        job_id="test-handler",
        job_name="测试处理专员",
        responsibilities=["处理测试请求"],
    )


class TestSaveAndLoadExamResult:
    def test_round_trip(self):
        result = ExamResult(exam_id="e-001", worker_id="w-001", job_id="test")
        _save_exam_result(result)
        loaded = load_exam_result("e-001")
        assert loaded.exam_id == "e-001"
        assert loaded.worker_id == "w-001"

    def test_load_not_found(self):
        with pytest.raises(FileNotFoundError, match="Exam result not found"):
            load_exam_result("nonexistent")


class TestExamRunner:
    @pytest.mark.anyio
    @patch("deerflow.factory.exam_runner.Examiner")
    @patch("deerflow.factory.exam_runner.Judge")
    @patch("deerflow.factory.exam_runner.load_job")
    async def test_run_exam_flow(self, mock_load_job, MockJudge, MockExaminer):
        # Setup mocks
        mock_load_job.return_value = _make_job()

        mock_examiner = MockExaminer.return_value
        mock_examiner.generate_exam.return_value = [
            ExamCase(case_id="r-001", category="routine", scenario="Test", expected_behavior="Handle"),
        ]
        mock_examiner.play_scenario.return_value = Message(role="examiner", content="")

        mock_judge = MockJudge.return_value
        mock_judge.evaluate_case.return_value = CaseResult(
            case_id="r-001",
            category="routine",
            passed=True,
            weighted_score=85.0,
        )
        mock_judge.generate_report.return_value = ExamResult(
            exam_id="e-test",
            worker_id="w-001",
            job_id="test-handler",
            passed=True,
            scores=ExamScores(overall=85.0),
        )

        runner = ExamRunner()
        runner._examiner = mock_examiner
        runner._judge = mock_judge

        # Mock worker response
        with patch.object(runner, "_get_worker_response", new_callable=AsyncMock) as mock_response:
            mock_response.return_value = "I have processed the request."

            plan = ExamPlan(job_id="test-handler")
            result = await runner.run_exam(_make_worker(), plan)

        assert result.passed is True
        assert result.exam_id == "e-test"
        mock_examiner.generate_exam.assert_called_once()
        mock_judge.evaluate_case.assert_called_once()
        mock_judge.generate_report.assert_called_once()
