"""Digital Worker Factory — unified entry point for the five-step pipeline.

Step 1: Job Description → save & L0
Step 2: Capability analysis
Step 3: Agent assembly → L1
Step 4: Exam → L2 → L3 (or rework)
Step 5: Return onboarded worker status
"""

from __future__ import annotations

import logging

from deerflow.factory.assembler import AgentAssembler, CapabilityAnalyzer
from deerflow.factory.exam_runner import ExamRunner
from deerflow.factory.job_store import save_job
from deerflow.factory.models import (
    AssemblyResult,
    CapabilityMatrix,
    ExamPlan,
    ExamResult,
    JobDescription,
    WorkerStatus,
)
from deerflow.factory.onboarding import OnboardingManager

logger = logging.getLogger(__name__)

MAX_REWORK_ROUNDS = 3


class DigitalWorkerFactory:
    """Orchestrates the complete five-step digital worker production pipeline."""

    def __init__(self) -> None:
        self._analyzer = CapabilityAnalyzer()
        self._assembler = AgentAssembler()
        self._exam_runner = ExamRunner()
        self._onboarding = OnboardingManager()

    # ── Full pipeline ────────────────────────────────────────────────────

    async def create_worker(
        self,
        job: JobDescription,
        exam_plan: ExamPlan | None = None,
    ) -> WorkerStatus:
        """Execute the full five-step pipeline.

        Args:
            job: The job description to create a worker for.
            exam_plan: Optional custom exam plan. Defaults to standard plan.

        Returns:
            The final WorkerStatus (L3 if passed, lower if failed after max rework).
        """
        # Step 1: Save JD
        save_job(job)
        logger.info("Step 1: Saved job description %s", job.job_id)

        # Step 2: Capability analysis
        capabilities = self._analyzer.analyze(job)
        logger.info("Step 2: Generated capability matrix for %s", job.job_id)

        # Step 3: Assembly
        assembly = self._assembler.assemble(job, capabilities)
        worker = WorkerStatus(
            worker_id=assembly.worker_id,
            job_id=job.job_id,
            agent_name=assembly.agent_name,
            level="L1",
        )
        self._onboarding.save_worker(worker)
        logger.info("Step 3: Assembled agent %s → L1", assembly.agent_name)

        # Step 4: Exam
        if exam_plan is None:
            exam_plan = ExamPlan(job_id=job.job_id)

        worker.level = "L2"
        self._onboarding.save_worker(worker)

        exam_result = await self._exam_runner.run_exam(worker, exam_plan)
        worker.exam_history.append(exam_result.exam_id)

        if exam_result.passed:
            worker.level = "L3"
            self._onboarding.save_worker(worker)
            logger.info("Step 4: Exam passed → L3")
        else:
            logger.info("Step 4: Exam failed, attempting rework")
            worker = await self._rework_loop(worker, exam_result, exam_plan)

        # Step 5: Return final status
        logger.info(
            "Step 5: Worker %s final level: %s",
            worker.worker_id,
            worker.level,
        )
        return worker

    # ── Individual steps (for API use) ───────────────────────────────────

    def analyze_job(self, job: JobDescription) -> CapabilityMatrix:
        """Step 2 only: analyze a job and return the capability matrix."""
        return self._analyzer.analyze(job)

    def assemble_agent(
        self,
        job: JobDescription,
        capabilities: CapabilityMatrix | None = None,
    ) -> AssemblyResult:
        """Step 3 only: assemble an agent from JD + capabilities."""
        return self._assembler.assemble(job, capabilities)

    async def run_exam(
        self,
        worker_id: str,
        exam_plan: ExamPlan | None = None,
    ) -> ExamResult:
        """Step 4 only: run an exam for an existing worker."""
        worker = self._onboarding.get_worker(worker_id)
        if exam_plan is None:
            exam_plan = ExamPlan(job_id=worker.job_id)
        return await self._exam_runner.run_exam(worker, exam_plan)

    # ── Rework ───────────────────────────────────────────────────────────

    async def rework(
        self,
        worker_id: str,
        diagnosis: str,
    ) -> WorkerStatus:
        """Rework a failed worker: re-assemble and re-examine.

        Args:
            worker_id: The worker to rework.
            diagnosis: Diagnosis from the previous exam failure.

        Returns:
            Updated WorkerStatus.
        """
        worker = self._onboarding.get_worker(worker_id)
        if worker.rework_count >= MAX_REWORK_ROUNDS:
            raise ValueError(f"Worker {worker_id} has reached max rework rounds ({MAX_REWORK_ROUNDS})")

        from deerflow.factory.job_store import load_job

        job = load_job(worker.job_id)

        # Re-assemble with updated context
        assembly = self._assembler.assemble(job)
        worker.agent_name = assembly.agent_name
        worker.rework_count += 1
        worker.level = "L1"
        self._onboarding.save_worker(worker)

        # Re-examine
        worker.level = "L2"
        self._onboarding.save_worker(worker)

        exam_plan = ExamPlan(job_id=worker.job_id)
        exam_result = await self._exam_runner.run_exam(worker, exam_plan)
        worker.exam_history.append(exam_result.exam_id)

        if exam_result.passed:
            worker.level = "L3"
        self._onboarding.save_worker(worker)
        return worker

    async def _rework_loop(
        self,
        worker: WorkerStatus,
        last_result: ExamResult,
        plan: ExamPlan,
    ) -> WorkerStatus:
        """Attempt rework up to MAX_REWORK_ROUNDS times."""
        for attempt in range(MAX_REWORK_ROUNDS):
            worker.rework_count += 1
            logger.info(
                "Rework attempt %d/%d for %s",
                attempt + 1,
                MAX_REWORK_ROUNDS,
                worker.worker_id,
            )

            from deerflow.factory.job_store import load_job

            job = load_job(worker.job_id)

            # Re-assemble
            assembly = self._assembler.assemble(job)
            worker.agent_name = assembly.agent_name
            worker.level = "L1"
            self._onboarding.save_worker(worker)

            # Re-examine
            worker.level = "L2"
            self._onboarding.save_worker(worker)

            last_result = await self._exam_runner.run_exam(worker, plan)
            worker.exam_history.append(last_result.exam_id)

            if last_result.passed:
                worker.level = "L3"
                self._onboarding.save_worker(worker)
                logger.info("Rework succeeded on attempt %d", attempt + 1)
                return worker

        # All rework attempts failed
        self._onboarding.save_worker(worker)
        logger.warning(
            "Worker %s failed after %d rework attempts",
            worker.worker_id,
            MAX_REWORK_ROUNDS,
        )
        return worker
