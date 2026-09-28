"""Harmony (gpt-oss) prompt-safety regression tests.

gpt-oss-20b is served through llama-server's harmony chat template: the
FIRST system message is rendered as the ``# Instructions`` developer block,
so that is the only reliable place to tell the model "do not make
unnecessary fatal changes".  These tests pin:

  * ``is_harmony_model`` detection (routing label AND cached server id),
  * ``apply_harmony_prompt`` shaping (prepend/merge/idempotent/pass-through),
  * the payload wiring in ``LlamaProvider._build_payload``,
  * the safety rules in ``agent._SYSTEM_PROMPT`` itself.
"""
from agent import _SYSTEM_PROMPT
from agent_core.llm.harmony import (
    HARMONY_GIT_ADDENDUM,
    HARMONY_SAFETY_ADDENDUM,
    apply_harmony_prompt,
    is_harmony_model,
)
from agent_core.llm.llama_provider import LlamaProvider


class TestIsHarmonyModel:
    def test_gpt_oss_bare_id_detected(self):
        assert is_harmony_model("gpt-oss-20b-MXFP4")

    def test_gpt_oss_with_routing_prefix_detected(self):
        assert is_harmony_model("llama/gpt-oss-20b-MXFP4")

    def test_short_routing_label_detected(self):
        assert is_harmony_model("llama/oss-20b")

    def test_other_family_not_detected(self):
        assert not is_harmony_model("qwen3-coder-30b-a3b-instruct")

    def test_empty_name_not_detected(self):
        assert not is_harmony_model("")


class TestApplyHarmonyPrompt:
    def test_prepends_system_message_when_history_starts_with_user(self):
        msgs = apply_harmony_prompt(
            [{"role": "user", "content": "hi"}], "gpt-oss-20b")
        assert msgs[0]["role"] == "system"
        assert HARMONY_SAFETY_ADDENDUM.strip() in msgs[0]["content"]
        assert msgs[1] == {"role": "user", "content": "hi"}

    def test_merges_into_existing_leading_system_message(self):
        msgs = apply_harmony_prompt(
            [{"role": "system", "content": "BASE"},
             {"role": "user", "content": "hi"}],
            "gpt-oss-20b")
        assert msgs[0]["role"] == "system"
        assert "BASE" in msgs[0]["content"]
        assert HARMONY_SAFETY_ADDENDUM.strip() in msgs[0]["content"]
        assert len(msgs) == 2

    def test_is_idempotent(self):
        once = apply_harmony_prompt(
            [{"role": "system", "content": "BASE"}], "gpt-oss-20b")
        twice = apply_harmony_prompt(once, "gpt-oss-20b")
        content = twice[0]["content"]
        assert content.count(HARMONY_SAFETY_ADDENDUM.strip()) == 1
        assert content.count(HARMONY_GIT_ADDENDUM.strip()) == 1

    def test_addendum_carries_the_full_commit_sequence(self):
        text = HARMONY_GIT_ADDENDUM.lower()
        assert "commit changes" in text
        assert "never run `git push` alone" in text
        assert "add -a" in text
        assert "commit -m" in text

    def test_prepended_system_carries_git_addendum(self):
        msgs = apply_harmony_prompt(
            [{"role": "user", "content": "commit changes"}], "gpt-oss-20b")
        content = msgs[0]["content"]
        assert HARMONY_GIT_ADDENDUM.strip() in content
        assert HARMONY_SAFETY_ADDENDUM.strip() in content

    def test_non_harmony_model_leaves_messages_untouched(self):
        original = [{"role": "user", "content": "hi"}]
        out = apply_harmony_prompt(original, "qwen3-coder-30b-a3b-instruct")
        assert out == original

    def test_addendum_forbids_fatal_changes(self):
        text = HARMONY_SAFETY_ADDENDUM.lower()
        assert "fatal" in text
        assert "rm -rf" in text
        assert "git reset --hard" in text


class TestLlamaPayloadWiring:
    def test_payload_for_gpt_oss_carries_safety_addendum(self):
        prov = LlamaProvider(model_name="llama/oss-20b",
                             api_url="http://h/v1")
        prov._cached_server_model_id = "gpt-oss-20b-MXFP4"
        payload = prov._build_payload([{"role": "user", "content": "hi"}])
        first = payload["messages"][0]
        assert first["role"] == "system"
        assert HARMONY_SAFETY_ADDENDUM.strip() in first["content"]

    def test_payload_for_other_model_has_no_addendum(self):
        prov = LlamaProvider(model_name="llama/qwen3-coder-30b-a3b-instruct",
                             api_url="http://h/v1")
        prov._cached_server_model_id = "qwen3-coder-30b-a3b-instruct"
        payload = prov._build_payload([{"role": "user", "content": "hi"}])
        joined = " ".join(str(m.get("content") or "") for m in payload["messages"])
        assert HARMONY_SAFETY_ADDENDUM.strip() not in joined


class TestSystemPromptSafetyRules:
    def test_system_prompt_forbids_fatal_changes(self):
        text = _SYSTEM_PROMPT.lower()
        assert "fatal" in text

    def test_system_prompt_names_the_dangerous_commands(self):
        text = _SYSTEM_PROMPT.lower()
        assert "rm -rf" in text
        assert "git reset --hard" in text
        assert "force push" in text

    def test_system_prompt_demands_minimal_targeted_edits(self):
        assert "targeted edit" in _SYSTEM_PROMPT.lower()
