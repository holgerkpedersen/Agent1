Task 1: `agent1/types.py` — Define shared enums (ThreatCategory, SeverityLevel, RemediationStatus, PipelineStage) and TypedDicts for vulnerability reports and summaries. Include `from __future__ import annotations`. Max ~60 lines.

Task 2: `agent1/safety_config.py` — Define frozen SafetyConfig dataclass with validated fields (thresholds, allowed capabilities, timeout limits). Factory load method. Validates config at load time. Max ~80 lines.

Task 3: `agent1/threat_model.py` — Define ThreatProfile frozen dataclass and registry of known threat profiles for self-improvement systems. Lookup function get_threat_profile. Max ~70 lines.

Task 4: `agent1/vulnerability.py` — Define frozen Vulnerability dataclass with id, category, severity, description, affected_component, evidence. Define VulnerabilityReport with tuple of vulnerabilities and summary metadata. Factory create_vulnerability with validation. Max ~80 lines.

Task 5: `agent1/agent_state.py` — Define AgentState TypedDict with code_snapshot, policy_version, reward_function_ref, capability_level, improvement_history. Serialization helpers to/from JSON. Max ~50 lines.

Task 6: `agent1/scanner_base.py` — Define abstract VulnerabilityScanner base class with scan interface, logging hooks, and result aggregation helper. Enforces Open/Closed and Dependency Inversion SOLID principles. Max ~60 lines.

Task 7: `agent1/remediation_base.py` — Define frozen RemediationResult dataclass and abstract RemediationStrategy base class with apply method signature, dry-run support, and rollback capability. Max ~70 lines.

Task 8: `agent1/prompt_injection_scanner.py` — Implement VulnerabilityScanner for prompt injection via self-modification paths. Detects unsanitized input in code generation prompts, escape-sequence injection, indirect prompt injection through stored data. Returns VulnerabilityReport. Max ~120 lines.

Task 9: `agent1/reward_hacking_scanner.py` — Implement VulnerabilityScanner for reward hacking and reward tampering. Detects self-reward manipulation, reward function overfitting, proxy metric exploitation. Max ~110 lines.

Task 10: `agent1/capability_escalation_scanner.py` — Implement VulnerabilityScanner for capability escalation risks. Detects unauthorized capability grants, recursive improvement loops, sandbox escape vectors. Max ~120 lines.

Task 11: `agent1/prompt_guard_remediator.py` — Implement RemediationStrategy for prompt injection vulnerabilities. Applies input sanitization, output validation, and prompt separation policies. Returns RemediationResult with before/after state diff. Max ~100 lines.

Task 12: `agent1/reward_clamp_remediator.py` — Implement RemediationStrategy for reward hacking vulnerabilities. Applies reward clamping, monotonicity checks, and proxy metric guards. Max ~90 lines.

Task 13: `agent1/capability_limit_remediator.py` — Implement RemediationStrategy for capability escalation. Enforces capability caps, improvement rate limits, and sandbox boundaries. Max ~100 lines.

Task 14: `agent1/safety_verifier.py` — Define SafetyVerifier class that runs post-remediation verification. Runs regression checks against threat model profiles, validates remediations don't introduce new vulnerabilities (dual-check). Returns VerificationResult with pass/fail and confidence score. Max ~100 lines.

Task 15: `agent1/audit_logger.py` — Define AuditLogger class for append-only audit trail of all pipeline operations. Logs scan results, remediation actions, verification outcomes. Supports JSON structured output. Max ~80 lines.

Task 16: `agent1/pipeline_orchestrator.py` — Define SelfImprovementPipeline class orchestrating full pipeline (SOLID: Single Responsibility). Stages: scan → remediate → verify → audit. Supports dry-run mode, parallel scanner execution, and rollback on verification failure. Max ~130 lines.

Task 17: `agent1/main.py` — CLI entry point using argparse. Loads config, runs pipeline, outputs structured JSON report. Max ~60 lines.

Task 18: `pyproject.toml` — Project metadata, dependencies (mypy, pytest, pydantic), mypy configuration with --strict flag enabled for type-checking validation across all modules.