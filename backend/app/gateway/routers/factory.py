"""Gateway API router for the Digital Worker Factory.

Exposes REST endpoints for job management, worker creation,
exam execution, and the full five-step pipeline.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from deerflow.factory.factory import DigitalWorkerFactory
from deerflow.factory.job_store import delete_job, list_jobs, load_job, save_job
from deerflow.factory.models import (
    CapabilityMatrix,
    ExamPlan,
    ExamResult,
    JobDescription,
    WorkerStatus,
)
from deerflow.factory.onboarding import OnboardingManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/factory", tags=["factory"])

# ── Singletons ───────────────────────────────────────────────────────────

_factory = DigitalWorkerFactory()
_onboarding = OnboardingManager()


# ── Request / Response models ────────────────────────────────────────────


class JobListResponse(BaseModel):
    jobs: list[JobDescription]


class WorkerListResponse(BaseModel):
    workers: list[WorkerStatus]


class AnalyzeResponse(BaseModel):
    capabilities: CapabilityMatrix


class AssembleResponse(BaseModel):
    worker: WorkerStatus


class ExamResponse(BaseModel):
    result: ExamResult


class ReworkRequest(BaseModel):
    diagnosis: str = Field(default="", description="Diagnosis from previous failure")


class CreateWorkerRequest(BaseModel):
    job: JobDescription
    exam_plan: ExamPlan | None = Field(default=None, description="Optional custom exam plan")


class CreateWorkerResponse(BaseModel):
    worker: WorkerStatus


# ── Job endpoints ────────────────────────────────────────────────────────


@router.get("/jobs", response_model=JobListResponse, summary="List all jobs")
async def list_jobs_endpoint() -> JobListResponse:
    """List all stored job descriptions."""
    return JobListResponse(jobs=list_jobs())


@router.post("/jobs", response_model=JobDescription, status_code=201, summary="Create a job")
async def create_job_endpoint(job: JobDescription) -> JobDescription:
    """Create or update a job description."""
    save_job(job)
    return job


@router.get("/jobs/{job_id}", response_model=JobDescription, summary="Get a job")
async def get_job_endpoint(job_id: str) -> JobDescription:
    """Get a job description by ID."""
    try:
        return load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.delete("/jobs/{job_id}", status_code=204, summary="Delete a job")
async def delete_job_endpoint(job_id: str) -> None:
    """Delete a job description."""
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


# ── Analysis & Assembly endpoints ────────────────────────────────────────


@router.post(
    "/jobs/{job_id}/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze job capabilities (Step 2)",
)
async def analyze_job_endpoint(job_id: str) -> AnalyzeResponse:
    """Run capability analysis on a job description."""
    try:
        job = load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    try:
        capabilities = _factory.analyze_job(job)
    except Exception as e:
        logger.exception("Capability analysis failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=str(e))

    return AnalyzeResponse(capabilities=capabilities)


@router.post(
    "/jobs/{job_id}/assemble",
    response_model=AssembleResponse,
    summary="Assemble agent (Step 3)",
)
async def assemble_job_endpoint(job_id: str) -> AssembleResponse:
    """Assemble an agent from a job description."""
    try:
        job = load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    try:
        assembly = _factory.assemble_agent(job)
    except Exception as e:
        logger.exception("Assembly failed for job %s", job_id)
        raise HTTPException(status_code=500, detail=str(e))

    worker = WorkerStatus(
        worker_id=assembly.worker_id,
        job_id=job_id,
        agent_name=assembly.agent_name,
        level="L1",
    )
    _onboarding.save_worker(worker)

    return AssembleResponse(worker=worker)


# ── Worker endpoints ─────────────────────────────────────────────────────


@router.get("/workers", response_model=WorkerListResponse, summary="List all workers")
async def list_workers_endpoint() -> WorkerListResponse:
    """List all digital workers."""
    return WorkerListResponse(workers=_onboarding.list_workers())


@router.get("/workers/{worker_id}", response_model=WorkerStatus, summary="Get worker status")
async def get_worker_endpoint(worker_id: str) -> WorkerStatus:
    """Get a worker's current status."""
    try:
        return _onboarding.get_worker(worker_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")


# ── Exam endpoints ───────────────────────────────────────────────────────


@router.post(
    "/workers/{worker_id}/exam",
    response_model=ExamResponse,
    summary="Start exam (Step 4)",
)
async def start_exam_endpoint(
    worker_id: str,
    plan: ExamPlan | None = None,
) -> ExamResponse:
    """Start an exam for a worker."""
    try:
        result = await _factory.run_exam(worker_id, plan)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Exam failed for worker %s", worker_id)
        raise HTTPException(status_code=500, detail=str(e))

    return ExamResponse(result=result)


@router.get(
    "/workers/{worker_id}/exam/{exam_id}",
    response_model=ExamResult,
    summary="Get exam result",
)
async def get_exam_result_endpoint(worker_id: str, exam_id: str) -> ExamResult:
    """Get a specific exam result."""
    from deerflow.factory.exam_runner import load_exam_result

    try:
        result = load_exam_result(exam_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Exam not found: {exam_id}")

    if result.worker_id != worker_id:
        raise HTTPException(status_code=404, detail="Exam does not belong to this worker")

    return result


# ── Rework & Promote endpoints ───────────────────────────────────────────


@router.post(
    "/workers/{worker_id}/rework",
    response_model=WorkerStatus,
    summary="Rework a failed worker",
)
async def rework_endpoint(worker_id: str, request: ReworkRequest) -> WorkerStatus:
    """Trigger a rework cycle for a failed worker."""
    try:
        return await _factory.rework(worker_id, request.diagnosis)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Rework failed for worker %s", worker_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/workers/{worker_id}/promote",
    response_model=WorkerStatus,
    summary="Manually promote a worker",
)
async def promote_endpoint(worker_id: str) -> WorkerStatus:
    """Manually promote a worker to the next level."""
    try:
        return _onboarding.promote(worker_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Full pipeline endpoint ───────────────────────────────────────────────


@router.post(
    "/create",
    response_model=CreateWorkerResponse,
    summary="One-click worker creation (full pipeline)",
)
async def create_worker_endpoint(request: CreateWorkerRequest) -> CreateWorkerResponse:
    """Execute the complete five-step pipeline to create a digital worker."""
    try:
        worker = await _factory.create_worker(request.job, request.exam_plan)
    except Exception as e:
        logger.exception("Worker creation failed")
        raise HTTPException(status_code=500, detail=str(e))

    return CreateWorkerResponse(worker=worker)
