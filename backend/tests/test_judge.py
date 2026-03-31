"""Tests for deerflow.factory.judge — evaluation and reporting."""

import json
from unittest.mock import MagicMock, patch

from deerflow.factory.judge import Judge
from deerflow.factory.models import CaseResult, ExamCase, ExamPlan, Message


def _make_case(category="routine") -> ExamCase:
    return ExamCase(
        case_id="test-001",
        category=category,
        scenario="Test scenario",
        expected_behavior="Handle correctly",
    )


def _make_conversation() -> list[Message]:
    return [
        Message(role="examiner", content="Please handle this request."),
        Message(role="worker", content="I have processed the request."),
    ]


def _mock_evaluation_json(passed=True, safety_score=100) -> str:
    return json.dumps(
        {
            "passed": passed,
            "scores": [
                {"dimension": "judgment_accuracy", "score": 85, "comment": "Good"},
                {"dimension": "reasoning_quality", "score": 80, "comment": "OK"},
                {"dimension": "reply_quality", "score": 75, "comment": "Fine"},
                {"dimension": "tool_usage", "score": 70, "comment": "Adequate"},
                {"dimension": "boundary_awareness", "score": 90, "comment": "Good"},
                {"dimension": "safety_compliance", "score": safety_score, "comment": "Compliant"},
            ],
            "comment": "Overall assessment",
        }
    )


class TestJudgeEvaluateCase:
    @patch("deerflow.models.factory.create_chat_model")
    def test_evaluate_passing_case(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=_mock_evaluation_json())
        mock_create_model.return_value = mock_model

        judge = Judge()
        result = judge.evaluate_case(_make_case(), _make_conversation())

        assert isinstance(result, CaseResult)
        assert result.passed is True
        assert result.case_id == "test-001"
        assert len(result.dimension_scores) == 6
        assert result.weighted_score > 0

    @patch("deerflow.models.factory.create_chat_model")
    def test_evaluate_failing_case(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=_mock_evaluation_json(passed=False, safety_score=0))
        mock_create_model.return_value = mock_model

        judge = Judge()
        result = judge.evaluate_case(_make_case("red_line"), _make_conversation())

        assert result.passed is False

    @patch("deerflow.models.factory.create_chat_model")
    def test_evaluate_invalid_json_returns_failure(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="not json")
        mock_create_model.return_value = mock_model

        judge = Judge()
        result = judge.evaluate_case(_make_case(), _make_conversation())

        assert result.passed is False
        assert "evaluation failed" in result.comment.lower()


class TestJudgeGenerateReport:
    def test_empty_results(self):
        judge = Judge()
        plan = ExamPlan(job_id="test")
        report = judge.generate_report("e-001", "w-001", "test", [], plan)
        assert report.passed is False
        assert report.diagnosis is not None

    def test_all_passing(self):
        judge = Judge()
        plan = ExamPlan(job_id="test")
        results = [
            CaseResult(
                case_id="r-001",
                category="routine",
                passed=True,
                weighted_score=85.0,
                dimension_scores=[],
            ),
            CaseResult(
                case_id="rl-001",
                category="red_line",
                passed=True,
                weighted_score=80.0,
                dimension_scores=[],
            ),
        ]
        report = judge.generate_report("e-001", "w-001", "test", results, plan)
        assert report.passed is True
        assert report.scores.overall == 82.5

    @patch("deerflow.models.factory.create_chat_model")
    def test_red_line_failure_causes_overall_fail(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content='{"diagnosis": "Red line violated", "remedy_plan": "Fix compliance"}')
        mock_create_model.return_value = mock_model

        judge = Judge()
        plan = ExamPlan(job_id="test")
        results = [
            CaseResult(case_id="r-001", category="routine", passed=True, weighted_score=90.0),
            CaseResult(case_id="rl-001", category="red_line", passed=False, weighted_score=80.0),
        ]
        # Red line failure → overall fail even with good scores
        report = judge.generate_report("e-001", "w-001", "test", results, plan)
        assert report.passed is False
