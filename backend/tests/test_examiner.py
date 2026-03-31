"""Tests for deerflow.factory.examiner — exam case generation and scenario play."""

import json
from unittest.mock import MagicMock, patch

import pytest

from deerflow.factory.examiner import Examiner
from deerflow.factory.models import ExamCase, ExamPlan, JobDescription, JobScenarios


def _make_job() -> JobDescription:
    return JobDescription(
        job_id="test-handler",
        job_name="测试处理专员",
        responsibilities=["处理测试请求"],
        scenarios=JobScenarios(
            routine=["正常查询"],
            edge_cases=["超时场景"],
            red_lines=["尝试绕过权限"],
        ),
    )


def _mock_cases_json() -> str:
    return json.dumps(
        [
            {"case_id": "r-001", "category": "routine", "scenario": "Normal query", "expected_behavior": "Reply normally"},
            {"case_id": "e-001", "category": "edge_case", "scenario": "Timeout", "expected_behavior": "Handle gracefully"},
            {"case_id": "rl-001", "category": "red_line", "scenario": "Bypass attempt", "expected_behavior": "Refuse"},
        ]
    )


class TestExaminer:
    @patch("deerflow.models.factory.create_chat_model")
    def test_generate_exam_returns_cases(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=_mock_cases_json())
        mock_create_model.return_value = mock_model

        examiner = Examiner()
        plan = ExamPlan(job_id="test-handler")
        cases = examiner.generate_exam(_make_job(), plan)

        assert len(cases) == 3
        assert all(isinstance(c, ExamCase) for c in cases)
        assert cases[0].category == "routine"
        assert cases[2].category == "red_line"

    @patch("deerflow.models.factory.create_chat_model")
    def test_generate_exam_assigns_missing_ids(self, mock_create_model):
        data = [{"category": "routine", "scenario": "Test", "expected_behavior": "Handle"}]
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=json.dumps(data))
        mock_create_model.return_value = mock_model

        examiner = Examiner()
        plan = ExamPlan(job_id="test-handler")
        cases = examiner.generate_exam(_make_job(), plan)

        assert len(cases) == 1
        assert cases[0].case_id.startswith("case-")

    @patch("deerflow.models.factory.create_chat_model")
    def test_generate_exam_invalid_json_raises(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="not json")
        mock_create_model.return_value = mock_model

        examiner = Examiner()
        plan = ExamPlan(job_id="test-handler")
        with pytest.raises(ValueError, match="invalid JSON"):
            examiner.generate_exam(_make_job(), plan)

    @patch("deerflow.models.factory.create_chat_model")
    def test_play_scenario(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="I need more details about the order.")
        mock_create_model.return_value = mock_model

        examiner = Examiner()
        case = ExamCase(
            case_id="r-001",
            category="routine",
            scenario="Customer asks about refund",
            expected_behavior="Provide refund status",
        )
        msg = examiner.play_scenario(case, "Your refund is being processed.")
        assert msg.role == "examiner"
        assert "more details" in msg.content
