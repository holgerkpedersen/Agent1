# Coding Plan: AI Agent Self-Improvement Vulnerability Hardening System

## Architecture Overview

A modular, safety-first pipeline that identifies vulnerabilities in an AI agent's self-improvement loop and applies hardened remediations. Built on SOLID principles with strict mypy typing throughout.

---

## File Inventory (18 files)

### 1. `pyproject.toml`
- Project metadata, dependencies (mypy, pytest, pydantic, etc.)
- mypy configuration with strict mode enabled

### 2. `src/__init__.py`
- Package init; exports public API surface

### 3. `src/types.py`
- Shared enums: `ThreatCategory`, `SeverityLevel`, `RemediationStatus`, `PipelineStage`
- TypedDicts for vulnerability reports and remediation results
- Max ~60 lines, single responsibility: type definitions only

### 4. `src/config.py`
- `SafetyConfig` dataclass with validated fields (thresholds, allowed capabilities, timeout limits)
- Factory function `load_config(path: str) -> SafetyConfig`
- Validates config at load time; raises on invalid values
- Max ~80 lines

### 5. `src/threat_model.py`
- `ThreatProfile` dataclass: threat category, description, attack vector, mitigation strategy
- Registry of known threat profiles for self-improvement systems
- `get_threat_profile(category: ThreatCategory) -> ThreatProfile` lookup function
- Max ~70 lines

### 6. `src/vulnerability.py`
- `Vulnerability` dataclass: id, category, severity, description, affected_component, evidence
- `VulnerabilityReport` containing list of vulnerabilities + summary metadata
- Factory `create_vulnerability(...)` with validation
- Max ~80 lines

### 7. `src/scanner_base.py`
- Abstract base class `VulnerabilityScanner` (SOLID: Open/Closed, Dependency Inversion)
- Defines `scan(agent_state: AgentState) -> VulnerabilityReport` interface
- Includes logging hook and result aggregation helper
- Max ~60 lines

### 8. `src/agent_state.py`
- `AgentState` TypedDict: code_snapshot, policy_version, reward_function_ref, capability_level, improvement_history
- Serialization helpers (to/from JSON)
- Max ~50 lines

### 9. `src/prompt_injection_scanner.py`
- Implements `VulnerabilityScanner` for prompt injection via self-modification paths
- Detects: unsanitized input in code generation prompts, escape-sequence injection, indirect prompt injection through stored data
- Returns `VulnerabilityReport` with categorized findings
- Max ~120 lines

### 10. `src/reward_hacking_scanner.py`
- Implements `VulnerabilityScanner` for reward hacking / reward tampering
- Detects: self-reward manipulation, reward function overfitting, proxy metric exploitation
- Max ~110 lines

### 11. `src/capability_escalation_scanner.py`
- Implements `VulnerabilityScanner` for capability escalation risks
- Detects: unauthorized capability grants, recursive improvement loops, sandbox escape vectors
- Max ~120 lines

### 12. `src/remediation_base.py`
- Abstract base class `RemediationStrategy` (SOLID: Dependency Inversion)
- Defines `apply(vulnerability: Vulnerability, agent_state: AgentState) -> RemediationResult`
- Includes dry-run mode and rollback support
- Max ~70 lines

### 13. `src/prompt_guard_remediator.py`
- Implements `RemediationStrategy` for prompt injection vulnerabilities
- Applies input sanitization, output validation, and prompt separation policies
- Returns `RemediationResult` with before/after state diff
- Max ~100 lines

### 14. `src/reward_clamp_remediator.py`
- Implements `RemediationStrategy` for reward hacking vulnerabilities
- Applies reward clamping, monotonicity checks, and proxy metric guards
- Max ~90 lines

### 15. `src/capability_limit_remediator.py`
- Implements `RemediationStrategy` for capability escalation
- Enforces capability caps, improvement rate limits, and sandbox boundaries
- Max ~100 lines

### 16. `src/safety_verifier.py`
- `SafetyVerifier` class: post-remediation verification
- Runs regression checks against threat model profiles
- Validates that remediations don't introduce new vulnerabilities (dual-check)
- Returns `VerificationResult` with pass/fail and confidence score
- Max ~100 lines

### 17. `src/audit_logger.py`
- `AuditLogger` class: append-only audit trail for all pipeline operations
- Logs: scan results, remediation actions, verification outcomes
- Supports JSON structured output and file rotation
- Max ~80 lines

### 18. `src/pipeline_orchestrator.py`
- `SelfImprovementPipeline` class: orchestrates the full pipeline (SOLID: Single Responsibility)
- Stages: scan → remediate → verify → audit
- Supports dry-run mode, parallel scanner execution, and rollback on verification failure
- Max ~130 lines

### 19. `src/main.py`
- CLI entry point using argparse
- Loads config, runs pipeline, outputs structured JSON report
- Max ~60 lines

---

## Dependency Graph

```
main.py → pipeline_orchestrator.py
pipeline_orchestrator.py → scanner_base.py, remediation_base.py, safety_verifier.py, audit_logger.py
pipeline_orchestrator.py → agent_state.py, config.py, types.py
prompt_injection_scanner.py → scanner_base.py, vulnerability.py, types.py
reward_hacking_scanner.py → scanner_base.py, vulnerability.py, types.py
capability_escalation_scanner.py → scanner_base.py, vulnerability.py, types.py
prompt_guard_remediator.py → remediation_base.py, vulnerability.py, agent_state.py
reward_clamp_remediator.py → remediation_base.py, vulnerability.py, agent_state.py
capability_limit_remediator.py → remediation_base.py, vulnerability.py, agent_state.py
safety_verifier.py → threat_model.py, types.py, vulnerability.py
audit_logger.py → types.py
```

---

## Key Design Decisions

1. **Strategy Pattern** for scanners and remediations — new threat types plug in without modifying existing code (Open/Closed)
2. **Dependency Inversion** — pipeline depends on abstractions (`VulnerabilityScanner`, `RemediationStrategy`), not concretions
3. **Immutable Reports** — vulnerability reports are dataclasses with frozen fields where appropriate
4. **Dry-Run Support** — every remediation can be previewed before application
5. **Rollback Capability** — if verification fails post-remediation, the pipeline reverts to pre-remediation state
6. **Structured Audit Trail** — every action is logged for traceability and compliance

---

## mypy Strict Compliance Strategy

- All function signatures include explicit parameter and return types
- No `Any` types; use `object` or proper generics where needed
- `--strict` flag in pyproject.toml mypy config
- Every module has `from __future__ import annotations` for forward reference support
- Generic type variables used only with bounded constraints (no unbound TypeVars)

---

## Total Estimated Lines: ~1,400 lines across 19 files

All files respect the ≤150 line constraint. Each file has a single, well-defined responsibility aligned with SOLID principles.