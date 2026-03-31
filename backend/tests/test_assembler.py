"""Tests for deerflow.factory.assembler — capability analysis and agent assembly."""

import json
from unittest.mock import MagicMock, patch

import pytest

from deerflow.factory.assembler import AgentAssembler, CapabilityAnalyzer
from deerflow.factory.models import CapabilityMatrix, JobDescription


def _make_job() -> JobDescription:
    return JobDescription(
        job_id="test-handler",
        job_name="测试处理专员",
        responsibilities=["处理测试请求", "返回测试结果"],
    )


def _mock_capability_json() -> str:
    return json.dumps(
        {
            "job_id": "test-handler",
            "hard_skills": [{"id": "skill-1", "name": "Skill 1", "description": "Desc", "verification": "Check"}],
            "soft_skills": [],
            "red_line_skills": [],
        }
    )


@pytest.fixture(autouse=True)
def _use_tmp_paths(tmp_path, monkeypatch):
    from deerflow.config import paths

    monkeypatch.setattr(paths, "_paths", paths.Paths(base_dir=tmp_path))
    yield


class TestCapabilityAnalyzer:
    @patch("deerflow.models.factory.create_chat_model")
    def test_analyze_returns_matrix(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=_mock_capability_json())
        mock_create_model.return_value = mock_model

        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze(_make_job())

        assert isinstance(result, CapabilityMatrix)
        assert result.job_id == "test-handler"
        assert len(result.hard_skills) == 1

    @patch("deerflow.models.factory.create_chat_model")
    def test_analyze_strips_markdown_fences(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=f"```json\n{_mock_capability_json()}\n```")
        mock_create_model.return_value = mock_model

        analyzer = CapabilityAnalyzer()
        result = analyzer.analyze(_make_job())
        assert result.job_id == "test-handler"

    @patch("deerflow.models.factory.create_chat_model")
    def test_analyze_invalid_json_raises(self, mock_create_model):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="not json")
        mock_create_model.return_value = mock_model

        analyzer = CapabilityAnalyzer()
        with pytest.raises(ValueError, match="invalid JSON"):
            analyzer.analyze(_make_job())


class TestAgentAssembler:
    @patch("deerflow.models.factory.create_chat_model")
    def test_assemble_creates_agent_dir(self, mock_create_model, tmp_path):
        # Mock both capability analysis and SOUL generation
        cap_json = _mock_capability_json()
        mock_model = MagicMock()
        mock_model.invoke.side_effect = [
            MagicMock(content=cap_json),  # capability analysis
            MagicMock(content="# Test Agent\nYou are a test handler."),  # SOUL generation
        ]
        mock_create_model.return_value = mock_model

        assembler = AgentAssembler()
        result = assembler.assemble(_make_job())

        assert result.agent_name == "factory-test-handler"
        assert result.worker_id.startswith("worker-test-handler-")
        assert "Test Agent" in result.soul_md

        # Verify agent directory was created
        from deerflow.config.paths import get_paths

        agent_dir = get_paths().agent_dir("factory-test-handler")
        assert agent_dir.exists()
        assert (agent_dir / "config.yaml").exists()
        assert (agent_dir / "SOUL.md").exists()

    @patch("deerflow.models.factory.create_chat_model")
    def test_assemble_with_provided_capabilities(self, mock_create_model, tmp_path):
        cap = CapabilityMatrix(
            job_id="test-handler",
            hard_skills=[],
            soft_skills=[],
            red_line_skills=[],
        )
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="# SOUL\nDone.")
        mock_create_model.return_value = mock_model

        assembler = AgentAssembler()
        result = assembler.assemble(_make_job(), capabilities=cap)

        assert result.capabilities == cap
