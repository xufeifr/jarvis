"""Exam Runner — orchestrates the full exam flow.

Wires together Examiner (generates cases, plays scenarios),
the worker Agent (answers), and the Judge (scores).
Results are persisted to `.deer-flow/factory/exams/`.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from deerflow.config.paths import get_paths
from deerflow.factory.examiner import Examiner
from deerflow.factory.job_store import load_job
from deerflow.factory.judge import Judge
from deerflow.factory.models import (
    CaseResult,
    ExamCase,
    ExamPlan,
    ExamResult,
    Message,
    WorkerStatus,
)

logger = logging.getLogger(__name__)

MAX_CONVERSATION_TURNS = 3  # max back-and-forth per case


def _exams_dir() -> Path:
    d = get_paths().base_dir / "factory" / "exams"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_exam_result(result: ExamResult) -> Path:
    path = _exams_dir() / f"{result.exam_id}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_exam_result(exam_id: str) -> ExamResult:
    """Load an exam result from disk."""
    path = _exams_dir() / f"{exam_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Exam result not found: {exam_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExamResult(**data)


class ExamRunner:
    """Orchestrate a full exam: generate cases → run each → judge → report."""

    def __init__(self) -> None:
        self._examiner = Examiner()
        self._judge = Judge()

    async def run_exam(self, worker: WorkerStatus, plan: ExamPlan) -> ExamResult:
        """Execute the complete exam flow.

        1. Load job description
        2. Examiner generates cases
        3. For each case: examiner presents → worker answers → (optional multi-turn) → judge scores
        4. Judge aggregates into final report
        5. Persist result
        """
        exam_id = f"exam-{uuid.uuid4().hex[:8]}"
        job = load_job(worker.job_id)

        # Step 1: Generate exam cases
        cases = self._examiner.generate_exam(job, plan)
        logger.info("Generated %d exam cases for worker %s", len(cases), worker.worker_id)

        # Step 2: Run each case
        case_results: list[CaseResult] = []
        for case in cases:
            result = await self._run_single_case(case, worker)
            case_results.append(result)

        # Step 3: Judge generates final report
        exam_result = self._judge.generate_report(
            exam_id=exam_id,
            worker_id=worker.worker_id,
            job_id=worker.job_id,
            results=case_results,
            plan=plan,
        )

        # Step 4: Persist
        _save_exam_result(exam_result)
        logger.info(
            "Exam %s completed: passed=%s, score=%.1f",
            exam_id,
            exam_result.passed,
            exam_result.scores.overall,
        )

        return exam_result

    async def _run_single_case(
        self,
        case: ExamCase,
        worker: WorkerStatus,
    ) -> CaseResult:
        """Run a single exam case: present scenario → get worker response → judge."""
        conversation: list[Message] = []

        # Initial scenario presentation
        conversation.append(Message(role="examiner", content=case.scenario))

        # Get worker response
        worker_response = await self._get_worker_response(worker, case.scenario)
        conversation.append(Message(role="worker", content=worker_response))

        # Optional multi-turn conversation
        for _turn in range(MAX_CONVERSATION_TURNS - 1):
            followup = self._examiner.play_scenario(case, worker_response)
            if not followup.content.strip():
                break
            conversation.append(followup)

            worker_response = await self._get_worker_response(worker, followup.content)
            conversation.append(Message(role="worker", content=worker_response))

        # Judge evaluates
        result = self._judge.evaluate_case(case, conversation)
        return result

    async def _get_worker_response(
        self,
        worker: WorkerStatus,
        message: str,
    ) -> str:
        """Get a response from the worker agent.

        Uses the agent system to invoke the worker's configured agent.
        """
        from deerflow.models.factory import create_chat_model

        # Create a model configured for the worker's agent
        model = create_chat_model()

        # Load the agent's SOUL.md for system context
        from deerflow.config.agents_config import load_agent_soul

        soul = load_agent_soul(worker.agent_name)
        messages = []
        if soul:
            messages.append(("system", soul))
        messages.append(("human", message))

        response = model.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        return content.strip()
