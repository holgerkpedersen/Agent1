"""Tests for cross-file review detectors: collision, attribute, unwired."""
import tempfile
import os

import pytest

from agent_core.patterns import (
    detect_module_collisions,
    detect_attribute_errors,
    detect_unwired_modules,
)


class TestModuleCollisions:
    def test_similar_names_detected(self):
        findings = detect_module_collisions(
            ["agent_core/llm/lm_studio_provider.py"],
            existing_files=["agent_core/llm/lmstudio.py", "agent_core/llm/tool_loop.py"],
        )
        assert len(findings) == 1
        assert "lmstudio.py" in findings[0]["suggestion"]

    def test_distinct_names_not_flagged(self):
        findings = detect_module_collisions(
            ["agent_core/llm/model_profiles.py"],
            existing_files=["agent_core/llm/lmstudio.py", "agent_core/llm/tool_loop.py"],
        )
        assert len(findings) == 0

    def test_same_directory_different_names_passes(self):
        findings = detect_module_collisions(
            ["agent_core/commands/analyze_cmd.py"],
            existing_files=["agent_core/commands/workflow_cmd.py"],
        )
        assert len(findings) == 0

    def test_excludes_init_py(self):
        findings = detect_module_collisions(
            ["agent_core/__init__.py"],
            existing_files=["agent_core/__init__.py"],
        )
        assert len(findings) == 0

    def test_empty_input(self):
        assert detect_module_collisions([]) == []

    def test_very_similar_names_flagged(self):
        findings = detect_module_collisions(
            ["agent_core/user_manager.py"],
            existing_files=["agent_core/user_management.py"],
        )
        assert len(findings) == 1

    def test_shared_concept_token_flagged(self):
        """path_guard vs path_utils share the 'path' token — fuzzy ratio alone
        misses it, the token check must catch it."""
        findings = detect_module_collisions(
            ["agent_core/security/path_guard.py"],
            existing_files=["agent_core/security/path_utils.py"],
        )
        assert len(findings) == 1
        assert "path_utils.py" in findings[0]["suggestion"]

    def test_generic_tokens_do_not_flag(self):
        """Shared generic tokens (cmd/core/util) must NOT count as duplication."""
        findings = detect_module_collisions(
            ["agent_core/commands/analyze_cmd.py"],
            existing_files=["agent_core/commands/workflow_cmd.py"],
        )
        assert len(findings) == 0


class TestAttributeErrors:
    def test_missing_attribute_detected(self):
        # Create a temporary module with a class definition
        with tempfile.TemporaryDirectory() as td:
            mod_path = os.path.join(td, "some_module", "models.py")
            os.makedirs(os.path.dirname(mod_path))
            with open(mod_path, "w") as f:
                f.write("""
from dataclasses import dataclass
@dataclass
class ProfileMetadata:
    name: str = ""
    temperature: float = 0.7
    reasoning_budget: int = 0
""")
            gen_path = os.path.join(td, "some_module", "consumer.py")
            source_files = {
                "some_module/consumer.py": """
from some_module.models import ProfileMetadata
def use(p: ProfileMetadata):
    x = p.reasoning_effort  # BUG: should be reasoning_budget
    y = p.temperature  # OK
""",
            }
            findings = detect_attribute_errors(source_files, project_root=td)
            assert len(findings) >= 1
            assert "reasoning_effort" in findings[0]["suggestion"]

    def test_valid_attribute_not_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            mod_path = os.path.join(td, "lib", "types.py")
            os.makedirs(os.path.dirname(mod_path))
            with open(mod_path, "w") as f:
                f.write("""
class MyType:
    name: str = ""
    value: int = 0
""")
            source_files = {
                "consumer.py": """
from lib.types import MyType
def use(t: MyType):
    return t.name
""",
            }
            findings = detect_attribute_errors(source_files, project_root=td)
            assert len(findings) == 0

    def test_empty_input(self):
        assert detect_attribute_errors({}, project_root=".") == []


class TestUnwiredModules:
    def test_unwired_module_detected(self):
        with tempfile.TemporaryDirectory() as td:
            gen_file = os.path.join(td, "new_module", "orphan.py")
            os.makedirs(os.path.dirname(gen_file))
            with open(gen_file, "w") as f:
                f.write("def unused(): pass\n")
            # No other file imports this module
            os.makedirs(os.path.join(td, "existing"))
            with open(os.path.join(td, "existing", "real.py"), "w") as f:
                f.write("print('hello')\n")
            rel_gen = "new_module/orphan.py"
            findings = detect_unwired_modules([rel_gen], project_root=td)
            assert len(findings) == 1
            assert "not imported" in findings[0]["suggestion"]

    def test_wired_module_not_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            gen_file = os.path.join(td, "used_module", "helper.py")
            os.makedirs(os.path.dirname(gen_file))
            with open(gen_file, "w") as f:
                f.write("def helper(): pass\n")
            consumer = os.path.join(td, "consumer.py")
            with open(consumer, "w") as f:
                f.write("from used_module.helper import helper\n")
            rel_gen = "used_module/helper.py"
            findings = detect_unwired_modules([rel_gen], project_root=td)
            assert len(findings) == 0

    def test_empty_input(self):
        assert detect_unwired_modules([], project_root=".") == []
