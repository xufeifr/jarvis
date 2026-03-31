"""Pydantic data models for the Digital Worker Factory.

Defines schemas for job descriptions, capability matrices, exam plans,
exam results, and worker status — corresponding to the five-step
production pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

# ── Job Description (岗位说明书) ──────────────────────────────────────────


class JobInput(BaseModel):
    """An input that the worker receives."""

    name: str = Field(..., description="Input name, e.g. '用户申诉工单'")
    description: str = Field(default="", description="What this input contains")
    format: str = Field(default="text", description="Expected format: text / json / file / etc.")


class JobOutput(BaseModel):
    """An output the worker produces."""

    name: str = Field(..., description="Output name, e.g. '处理决定'")
    description: str = Field(default="", description="What this output contains")
    format: str = Field(default="text", description="Output format")


class JobPermissions(BaseModel):
    """Permission boundary — what the worker can and cannot do."""

    can_do: list[str] = Field(default_factory=list, description="Allowed actions")
    cannot_do: list[str] = Field(default_factory=list, description="Forbidden actions (red lines)")


class JobScenarios(BaseModel):
    """Scenario definitions for the job."""

    routine: list[str] = Field(default_factory=list, description="Common / happy-path scenarios")
    edge_cases: list[str] = Field(default_factory=list, description="Edge-case scenarios")
    red_lines: list[str] = Field(default_factory=list, description="Red-line scenarios that must be rejected")


class JobDescription(BaseModel):
    """Complete job description (岗位说明书)."""

    job_id: str = Field(..., description="Unique identifier, e.g. 'refund-appeal-handler'")
    job_name: str = Field(..., description="Human-readable name, e.g. '退款申诉处理专员'")
    responsibilities: list[str] = Field(default_factory=list, description="Core responsibilities")
    inputs: list[JobInput] = Field(default_factory=list, description="Input definitions")
    outputs: list[JobOutput] = Field(default_factory=list, description="Output definitions")
    permissions: JobPermissions = Field(default_factory=JobPermissions, description="Permission boundary")
    quality_bar: dict[str, str] = Field(default_factory=dict, description="Quality standards")
    scenarios: JobScenarios = Field(default_factory=JobScenarios, description="Scenario definitions")


# ── Capability Matrix (能力矩阵) ─────────────────────────────────────────


class Capability(BaseModel):
    """A single capability item."""

    id: str = Field(..., description="Unique capability id, e.g. 'policy-lookup'")
    name: str = Field(..., description="Display name")
    description: str = Field(default="", description="What this capability entails")
    verification: str = Field(default="", description="How to verify this capability")


class CapabilityMatrix(BaseModel):
    """Capability matrix derived from a job description."""

    job_id: str
    hard_skills: list[Capability] = Field(default_factory=list, description="Hard skills")
    soft_skills: list[Capability] = Field(default_factory=list, description="Soft skills")
    red_line_skills: list[Capability] = Field(default_factory=list, description="Red-line awareness")


# ── Exam Plan & Cases (考核方案) ─────────────────────────────────────────


class ExamComposition(BaseModel):
    """How many cases of each category to generate."""

    routine: int = Field(default=5, ge=0)
    edge_cases: int = Field(default=3, ge=0)
    red_line: int = Field(default=2, ge=0)


class ExamPlan(BaseModel):
    """Exam plan for a specific job."""

    job_id: str
    total_cases: int = Field(default=10, ge=1)
    composition: ExamComposition = Field(default_factory=ExamComposition)
    pass_criteria: dict[str, str] = Field(
        default_factory=lambda: {
            "overall_score": ">=70",
            "red_line_pass_rate": "100%",
        },
        description="Pass criteria",
    )
    instant_fail: list[str] = Field(
        default_factory=lambda: ["安全合规违规", "红线场景处理错误"],
        description="Instant-fail conditions",
    )


class ExamCase(BaseModel):
    """A single exam case."""

    case_id: str = Field(..., description="Unique case identifier")
    category: Literal["routine", "edge_case", "red_line"] = Field(..., description="Case category")
    scenario: str = Field(..., description="Scenario description for the examinee")
    expected_behavior: str = Field(default="", description="Expected correct behavior")


# ── Exam Results (考核结果) ──────────────────────────────────────────────


class Message(BaseModel):
    """A single conversation message."""

    role: Literal["examiner", "worker", "system"] = Field(..., description="Message sender role")
    content: str = Field(..., description="Message content")


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: str = Field(..., description="Dimension name")
    weight: float = Field(..., ge=0, le=1, description="Weight (0-1)")
    score: float = Field(..., ge=0, le=100, description="Score (0-100)")
    comment: str = Field(default="", description="Evaluator comment")


class CaseResult(BaseModel):
    """Result of evaluating a single exam case."""

    case_id: str
    category: Literal["routine", "edge_case", "red_line"]
    passed: bool
    conversation: list[Message] = Field(default_factory=list)
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    weighted_score: float = Field(default=0.0, ge=0, le=100)
    comment: str = Field(default="")


class ExamScores(BaseModel):
    """Aggregated scores across all dimensions."""

    overall: float = Field(default=0.0, ge=0, le=100, description="Overall weighted score")
    judgment_accuracy: float = Field(default=0.0, ge=0, le=100, description="判断正确性 (35%)")
    reasoning_quality: float = Field(default=0.0, ge=0, le=100, description="推理过程 (20%)")
    reply_quality: float = Field(default=0.0, ge=0, le=100, description="回复质量 (15%)")
    tool_usage: float = Field(default=0.0, ge=0, le=100, description="工具使用 (10%)")
    boundary_awareness: float = Field(default=0.0, ge=0, le=100, description="边界意识 (10%)")
    safety_compliance: float = Field(default=0.0, ge=0, le=100, description="安全合规 (10%)")


class ExamResult(BaseModel):
    """Complete exam result."""

    exam_id: str
    worker_id: str
    job_id: str = ""
    cases: list[CaseResult] = Field(default_factory=list)
    scores: ExamScores = Field(default_factory=ExamScores)
    passed: bool = False
    diagnosis: str | None = Field(default=None, description="Diagnosis report when failed")
    remedy_plan: str | None = Field(default=None, description="Remedy plan when failed")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


# ── Worker Status (数字员工状态) ─────────────────────────────────────────

WorkerLevel = Literal["L0", "L1", "L2", "L3", "L4", "L5", "L6"]


class WorkerStats(BaseModel):
    """Runtime statistics for a worker."""

    total_cases: int = Field(default=0, ge=0, description="Total cases processed")
    successful_cases: int = Field(default=0, ge=0, description="Successfully processed cases")
    failed_cases: int = Field(default=0, ge=0, description="Failed cases")
    accuracy_rate: float = Field(default=0.0, ge=0, le=1, description="Accuracy rate (0-1)")
    consecutive_success: int = Field(default=0, ge=0, description="Consecutive successes (no major error)")
    edge_case_accuracy: float = Field(default=0.0, ge=0, le=1, description="Edge-case accuracy rate")
    new_scenario_count: int = Field(default=0, ge=0, description="Number of new scenario types handled")


class WorkerStatus(BaseModel):
    """Persistent status of a digital worker."""

    worker_id: str = Field(..., description="Unique worker identifier")
    job_id: str = Field(..., description="Associated job description id")
    agent_name: str = Field(default="", description="Corresponding AgentConfig name")
    level: WorkerLevel = Field(default="L0", description="Current onboarding level")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    exam_history: list[str] = Field(default_factory=list, description="List of exam_id references")
    stats: WorkerStats = Field(default_factory=WorkerStats, description="Runtime statistics")
    rework_count: int = Field(default=0, ge=0, description="Number of rework iterations")


# ── Assembly Result (装配结果) ───────────────────────────────────────────


class AssemblyResult(BaseModel):
    """Result of the agent assembly step."""

    worker_id: str
    agent_name: str
    soul_md: str = Field(default="", description="Generated SOUL.md content")
    config_yaml: dict = Field(default_factory=dict, description="Generated agent config")
    capabilities: CapabilityMatrix | None = None


# ── Case Record (运行时案例记录) ─────────────────────────────────────────


class CaseRecord(BaseModel):
    """A single case processing record during production use."""

    case_id: str
    success: bool
    is_edge_case: bool = False
    is_new_scenario: bool = False
    error_type: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
