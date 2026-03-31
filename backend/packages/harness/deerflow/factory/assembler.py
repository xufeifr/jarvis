"""Capability analysis and agent assembly.

Step 2 (能力拆解): LLM analyses a JobDescription → CapabilityMatrix.
Step 3 (定向装配): JD + CapabilityMatrix → Agent config package.
"""

from __future__ import annotations

import json
import logging
import uuid

import yaml
from pydantic import ValidationError

from deerflow.config.agents_config import load_agent_config
from deerflow.config.paths import get_paths
from deerflow.factory.models import (
    AssemblyResult,
    CapabilityMatrix,
    JobDescription,
)

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────

_CAPABILITY_ANALYSIS_PROMPT = """\
You are an expert in job capability analysis.
Given the following job description (in JSON), decompose it into a capability matrix.

Job Description:
{jd_json}

Output a JSON object with these fields:
- job_id: same as input
- hard_skills: array of capabilities needed (id, name, description, verification)
- soft_skills: array of soft capabilities (id, name, description, verification)
- red_line_skills: array of red-line awareness items (id, name, description, verification)

Each capability must have:
- id: short kebab-case identifier
- name: human-readable name
- description: what this capability entails
- verification: how to verify the agent has this capability

Respond ONLY with valid JSON. No markdown fences.
"""

_SOUL_GENERATION_PROMPT = """\
You are an expert in designing AI agent system prompts.
Given the job description and capability matrix below, generate a SOUL.md document
that defines this agent's personality, responsibilities, quality standards, and iron rules.

Job Description:
{jd_json}

Capability Matrix:
{cap_json}

The SOUL.md should include:
1. Agent identity and role description
2. Core responsibilities (from the JD)
3. Quality standards (from quality_bar)
4. Iron rules (derived from permissions.cannot_do — these are absolute prohibitions)
5. Working guidelines

Output the SOUL.md content as plain markdown. No JSON wrapping.
"""


class CapabilityAnalyzer:
    """Analyze a job description and produce a capability matrix using LLM."""

    def analyze(self, job: JobDescription) -> CapabilityMatrix:
        """Generate a capability matrix from the job description.

        Uses ``create_chat_model()`` to call the LLM with structured output.
        """
        from deerflow.models.factory import create_chat_model

        model = create_chat_model()
        jd_json = job.model_dump_json(indent=2)
        prompt = _CAPABILITY_ANALYSIS_PROMPT.format(jd_json=jd_json)

        response = model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Strip markdown fences if present
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("LLM returned invalid JSON for capability analysis: %s", e)
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

        try:
            return CapabilityMatrix(**data)
        except ValidationError as e:
            logger.error("LLM output failed validation: %s", e)
            raise ValueError(f"LLM output failed validation: {e}") from e


class AgentAssembler:
    """Assemble an Agent configuration package from JD + capabilities."""

    def __init__(self) -> None:
        self._analyzer = CapabilityAnalyzer()

    def assemble(
        self,
        job: JobDescription,
        capabilities: CapabilityMatrix | None = None,
    ) -> AssemblyResult:
        """Generate an Agent configuration package.

        1. Generate SOUL.md using LLM
        2. Build config.yaml
        3. Create agent directory
        4. Return AssemblyResult with worker_id

        If *capabilities* is None, runs capability analysis first.
        """
        if capabilities is None:
            capabilities = self._analyzer.analyze(job)

        worker_id = f"worker-{job.job_id}-{uuid.uuid4().hex[:8]}"
        agent_name = f"factory-{job.job_id}"

        # Generate SOUL.md
        soul_md = self._generate_soul(job, capabilities)

        # Build agent config
        config_data = {
            "name": agent_name,
            "description": job.job_name,
        }

        # Create agent directory and files
        agent_dir = get_paths().agent_dir(agent_name)
        agent_dir.mkdir(parents=True, exist_ok=True)

        config_file = agent_dir / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        soul_file = agent_dir / "SOUL.md"
        soul_file.write_text(soul_md, encoding="utf-8")

        # Verify the agent loads correctly
        try:
            load_agent_config(agent_name)
        except Exception as e:
            logger.warning("Created agent but failed to load config: %s", e)

        return AssemblyResult(
            worker_id=worker_id,
            agent_name=agent_name,
            soul_md=soul_md,
            config_yaml=config_data,
            capabilities=capabilities,
        )

    def _generate_soul(self, job: JobDescription, capabilities: CapabilityMatrix) -> str:
        """Use LLM to generate SOUL.md content."""
        from deerflow.models.factory import create_chat_model

        model = create_chat_model()
        prompt = _SOUL_GENERATION_PROMPT.format(
            jd_json=job.model_dump_json(indent=2),
            cap_json=capabilities.model_dump_json(indent=2),
        )
        response = model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return content.strip()
