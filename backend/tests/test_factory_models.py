"""Tests for deerflow.factory.models — Pydantic schema validation."""

from deerflow.factory.models import (
    AssemblyResult,
    Capability,
    CapabilityMatrix,
    CaseRecord,
    CaseResult,
    DimensionScore,
    ExamCase,
    ExamComposition,
    ExamPlan,
    ExamResult,
    ExamScores,
    JobDescription,
    JobInput,
    JobOutput,
    JobPermissions,
    JobScenarios,
    Message,
    WorkerStats,
    WorkerStatus,
)


def _make_job(**overrides) -> JobDescription:
    defaults = {
        "job_id": "test-handler",
        "job_name": "测试处理专员",
        "responsibilities": ["处理测试请求"],
    }
    defaults.update(overrides)
    return JobDescription(**defaults)


class TestJobDescription:
    def test_minimal_job(self):
        job = _make_job()
        assert job.job_id == "test-handler"
        assert job.job_name == "测试处理专员"
        assert job.permissions.can_do == []
        assert job.scenarios.routine == []

    def test_full_job(self):
        job = _make_job(
            inputs=[JobInput(name="工单", description="用户工单", format="json")],
            outputs=[JobOutput(name="决定", description="处理决定")],
            permissions=JobPermissions(can_do=["查询"], cannot_do=["退款"]),
            quality_bar={"准确率": ">=95%"},
            scenarios=JobScenarios(
                routine=["正常查询"],
                edge_cases=["超时"],
                red_lines=["绕过权限"],
            ),
        )
        assert len(job.inputs) == 1
        assert job.inputs[0].format == "json"
        assert job.permissions.cannot_do == ["退款"]
        assert len(job.scenarios.red_lines) == 1

    def test_round_trip_json(self):
        job = _make_job()
        data = job.model_dump(mode="json")
        restored = JobDescription(**data)
        assert restored == job


class TestCapabilityMatrix:
    def test_empty_matrix(self):
        m = CapabilityMatrix(job_id="test")
        assert m.hard_skills == []
        assert m.soft_skills == []
        assert m.red_line_skills == []

    def test_with_capabilities(self):
        cap = Capability(id="c1", name="Cap 1", description="Desc", verification="Check")
        m = CapabilityMatrix(job_id="test", hard_skills=[cap])
        assert len(m.hard_skills) == 1
        assert m.hard_skills[0].id == "c1"


class TestExamPlan:
    def test_defaults(self):
        plan = ExamPlan(job_id="test")
        assert plan.total_cases == 10
        assert plan.composition.routine == 5
        assert plan.composition.edge_cases == 3
        assert plan.composition.red_line == 2

    def test_custom_plan(self):
        plan = ExamPlan(
            job_id="test",
            total_cases=20,
            composition=ExamComposition(routine=10, edge_cases=6, red_line=4),
        )
        assert plan.total_cases == 20


class TestExamCase:
    def test_case_creation(self):
        case = ExamCase(
            case_id="r-001",
            category="routine",
            scenario="Customer asks for refund status",
            expected_behavior="Look up order and provide status",
        )
        assert case.category == "routine"


class TestCaseResult:
    def test_with_scores(self):
        result = CaseResult(
            case_id="r-001",
            category="routine",
            passed=True,
            dimension_scores=[
                DimensionScore(dimension="judgment_accuracy", weight=0.35, score=90, comment="Good"),
            ],
            weighted_score=31.5,
        )
        assert result.passed
        assert result.weighted_score == 31.5


class TestExamResult:
    def test_defaults(self):
        result = ExamResult(exam_id="e-001", worker_id="w-001")
        assert result.passed is False
        assert result.scores.overall == 0.0
        assert result.diagnosis is None


class TestWorkerStatus:
    def test_defaults(self):
        w = WorkerStatus(worker_id="w-001", job_id="test")
        assert w.level == "L0"
        assert w.stats.total_cases == 0
        assert w.rework_count == 0

    def test_round_trip(self):
        w = WorkerStatus(worker_id="w-001", job_id="test", level="L3")
        data = w.model_dump(mode="json")
        restored = WorkerStatus(**data)
        assert restored.level == "L3"


class TestWorkerStats:
    def test_defaults(self):
        s = WorkerStats()
        assert s.accuracy_rate == 0.0
        assert s.consecutive_success == 0


class TestAssemblyResult:
    def test_basic(self):
        r = AssemblyResult(worker_id="w-001", agent_name="factory-test")
        assert r.soul_md == ""
        assert r.capabilities is None


class TestCaseRecord:
    def test_success_record(self):
        r = CaseRecord(case_id="c-001", success=True)
        assert r.is_edge_case is False

    def test_failed_record(self):
        r = CaseRecord(case_id="c-002", success=False, error_type="wrong_judgment")
        assert r.error_type == "wrong_judgment"


class TestMessage:
    def test_roles(self):
        for role in ("examiner", "worker", "system"):
            m = Message(role=role, content="hello")
            assert m.role == role


class TestExamScores:
    def test_defaults(self):
        s = ExamScores()
        assert s.overall == 0.0
        assert s.safety_compliance == 0.0
