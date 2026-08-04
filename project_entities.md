```python
from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple


# === Shared Enums (types.py) ===

class ThreatCategory(Enum):
    PROMPT_INJECTION = auto()
    REWARD_HACKING = auto()
    CAPABILITY_ESCALATION = auto()
    RECURSIVE_MODIFICATION = auto()
    SANDBOX_ESCAPE = auto()


class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RemediationStatus(Enum):
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


class PipelineStage(Enum):
    SCAN = "scan"
    REMEDIATE = "remediate"
    VERIFY = "verify"
    AUDIT = "audit"


# === TypedDicts (types.py) ===

class VulnerabilityReportSummary(TypedDict, total=False):
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    scan_timestamp: str
    agent_version: str


# === Config (config.py) ===

@dataclass(frozen=True)
class SafetyConfig:
    severity_threshold: SeverityLevel = SeverityLevel.MEDIUM
    max_concurrent_scanners: int = 4
    improvement_rate_limit: float = 1.0
    capability_cap: float = 100.0
    timeout_seconds: float = 300.0
    dry_run_enabled: bool = True
    rollback_on_failure: bool = True
    allowed_capabilities: List[str] = field(default_factory=list)
    reward_clamp_min: float = -1000.0
    reward_clamp_max: float = 1000.0

    @classmethod
    def load(cls, path: str) -> SafetyConfig:
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# === Threat Model (threat_model.py) ===

@dataclass(frozen=True)
class ThreatProfile:
    category: ThreatCategory
    description: str
    attack_vector: str
    mitigation_strategy: str
    detection_signatures: List[str] = field(default_factory=list)


_THREAT_REGISTRY: Dict[ThreatCategory, ThreatProfile] = {
    ThreatCategory.PROMPT_INJECTION: ThreatProfile(
        category=ThreatCategory.PROMPT_INJECTION,
        description="Malicious input crafted to manipulate agent behavior through self-modification prompts",
        attack_vector="Unsantized user input injected into code generation pipelines",
        mitigation_strategy="Input sanitization and output validation with prompt separation policies",
        detection_signatures=["escape_sequences", "indirect_injection", "prompt_leakage"],
    ),
    ThreatCategory.REWARD_HACKING: ThreatProfile(
        category=ThreatCategory.REWARD_HACKING,
        description="Agent manipulates its own reward function to achieve artificially high scores",
        attack_vector="Self-reward manipulation and proxy metric exploitation",
        mitigation_strategy="Reward clamping, monotonicity checks, and proxy metric guards",
        detection_signatures=["reward_spike", "proxy_overfitting", "self_reward"],
    ),
    ThreatCategory.CAPABILITY_ESCALATION: ThreatProfile(
        category=ThreatCategory.CAPABILITY_ESCALATION,
        description="Agent grants itself unauthorized capabilities beyond intended scope",
        attack_vector="Recursive improvement loops and sandbox escape vectors",
        mitigation_strategy="Capability caps, improvement rate limits, and sandbox boundaries",
        detection_signatures=["unauthorized_grant", "recursive_loop", "sandbox_breach"],
    ),
}


def get_threat_profile(category: ThreatCategory) -> ThreatProfile:
    profile = _THREAT_REGISTRY.get(category)
    if profile is None:
        raise ValueError(f"Unknown threat category: {category}")
    return profile


# === Agent State (agent_state.py) ===

class AgentState(TypedDict):
    code_snapshot: str
    policy_version: str
    reward_function_ref: str
    capability_level: float
    improvement_history: List[Dict[str, object]]


def serialize_agent_state(state: AgentState) -> str:
    return json.dumps(state, default=str)


def deserialize_agent_state(raw: str) -> AgentState:
    data = json.loads(raw)
    return AgentState(
        code_snapshot=data["code_snapshot"],
        policy_version=data["policy_version"],
        reward_function_ref=data["reward_function_ref"],
        capability_level=float(data["capability_level"]),
        improvement_history=list(data.get("improvement_history", [])),
    )


# === Vulnerability (vulnerability.py) ===

@dataclass(frozen=True)
class Vulnerability:
    id: str
    category: ThreatCategory
    severity: SeverityLevel
    description: str
    affected_component: str
    evidence: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class VulnerabilityReport:
    vulnerabilities: Tuple[Vulnerability, ...]
    summary: VulnerabilityReportSummary

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == SeverityLevel.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == SeverityLevel.HIGH)


def create_vulnerability(
    category: ThreatCategory,
    severity: SeverityLevel,
    description: str,
    affected_component: str,
    evidence: Optional[List[str]] = None,
) -> Vulnerability:
    if not description.strip():
        raise ValueError("Vulnerability description cannot be empty")
    if not affected_component.strip():
        raise ValueError("Affected component cannot be empty")
    return Vulnerability(
        id=str(uuid.uuid4()),
        category=category,
        severity=severity,
        description=description.strip(),
        affected_component=affected_component.strip(),
        evidence=list(evidence) if evidence is not None else [],
    )


# === Scanner Base (scanner_base.py) ===

class VulnerabilityScanner(ABC):
    def __init__(self, name: str) -> None:
        self._name = name
        self._logger = logging.getLogger(f"scanner.{name}")

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def scan(self, agent_state: AgentState) -> VulnerabilityReport:
        ...

    def _log_scan_start(self, agent_state: AgentState) -> None:
        self._logger.info(
            "Starting %s scan for policy_version=%s",
            self._name,
            agent_state["policy_version"],
        )

    def _aggregate_reports(self, reports: List[VulnerabilityReport]) -> VulnerabilityReport:
        all_vulns: List[Vulnerability] = []
        for report in reports:
            all_vulns.extend(report.vulnerabilities)
        summary: VulnerabilityReportSummary = {
            "total_vulnerabilities": len(all_vulns),
            "critical_count": sum(1 for v in all_vulns if v.severity == SeverityLevel.CRITICAL),
            "high_count": sum(1 for v in all_vulns if v.severity == SeverityLevel.HIGH),
        }
        return VulnerabilityReport(vulnerabilities=tuple(all_vulns), summary=summary)


# === Remediation Base (remediation_base.py) ===

@dataclass(frozen=True)
class RemediationResult:
    vulnerability_id: str
    status: RemediationStatus
    description: str
    before_state: Optional[AgentState] = None
    after_state: Optional[AgentState] = None
    rollback_available: bool = False


class RemediationStrategy(ABC):
    def __init__(self, name: str) -> None:
        self._name = name
        self._logger = logging.getLogger(f"remediation.{name}")

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def apply(
        self,
        vulnerability: Vulnerability,
        agent_state: AgentState,
        config: SafetyConfig,
    ) -> RemediationResult:
        ...

    def dry_run(
        self,
        vulnerability: Vulnerability,
        agent_state: AgentState,
        config: SafetyConfig,
    ) -> RemediationResult:
        return self.apply(vulnerability, agent_state, config)


# === Safety Verifier (safety_verifier.py) ===

@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    confidence_score: float
    findings: List[str] = field(default_factory=list)
    threat_profiles_checked: Tuple[ThreatProfile, ...] = field(default_factory=tuple)


class SafetyVerifier:
    def __init__(self, config: SafetyConfig) -> None:
        self._config = config

    def verify(
        self,
        agent_state: AgentState,
        report: VulnerabilityReport,
    ) -> VerificationResult:
        findings: List[str] = []
        profiles_checked: List[ThreatProfile] = []

        for vuln in report.vulnerabilities:
            profile = get_threat_profile(vuln.category)
            profiles_checked.append(profile)
            if vuln.severity == SeverityLevel.CRITICAL:
                findings.append(
                    f"Critical vulnerability {vuln.id} requires immediate attention"
                )

        passed = not any("requires immediate" in f for f in findings)
        confidence = 1.0 - (len(findings) * 0.1)
        confidence = max(0.0, min(1.0, confidence))

        return VerificationResult(
            passed=passed,
            confidence_score=round(confidence, 2),
            findings=findings,
            threat_profiles_checked=tuple(profiles_checked),
        )


# === Audit Logger (audit_logger.py) ===

class AuditLogger:
    def __init__(self, log_path: str) -> None:
        self._log_path = log_path
        self._logger = logging.getLogger("audit")
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def log_scan(self, scanner_name: str, report: VulnerabilityReport) -> None:
        entry: Dict[str, object] = {
            "event": "scan",
            "scanner": scanner_name,
            "vulnerabilities_found": len(report.vulnerabilities),
            "summary": dict(report.summary),
        }
        self._logger.info(json.dumps(entry))

    def log_remediation(self, strategy_name: str, result: RemediationResult) -> None:
        entry: Dict[str, object] = {
            "event": "remediation",
            "strategy": strategy_name,
            "status": result.status.value,
            "vulnerability_id": result.vulnerability_id,
        }
        self._logger.info(json.dumps(entry))

    def log_verification(self, result: VerificationResult) -> None:
        entry: Dict[str, object] = {
            "event": "verification",
            "passed": result.passed,
            "confidence": result.confidence_score,
            "findings_count": len(result.findings),
        }
        self._logger.info(json.dumps(entry))


# === Pipeline Orchestrator (pipeline_orchestrator.py) ===

class SelfImprovementPipeline:
    def __init__(
        self,
        scanners: Sequence[VulnerabilityScanner],
        remediations: Dict[ThreatCategory, RemediationStrategy],
        config: SafetyConfig,
        audit_logger: AuditLogger,
    ) -> None:
        self._scanners = list(scanners)
        self._remediations = dict(remediations)
        self._config = config
        self._audit_logger = audit_logger
        self._logger = logging.getLogger("pipeline")

    def run(
        self,
        agent_state: AgentState,
        dry_run: bool = True,
    ) -> Tuple[VulnerabilityReport, Optional[RemediationResult], VerificationResult]:
        scan_report = self._run_scans(agent_state)
        self._audit_logger.log_scan("pipeline", scan_report)

        if dry_run or not scan_report.vulnerabilities:
            verification = self._verify(agent_state, scan_report)
            return scan_report, None, verification

        remediation_result = self._run_remediation(scan_report, agent_state)
        self._audit_logger.log_remediation("pipeline", remediation_result)

        verification = self._verify(agent_state, scan_report)
        self._audit_logger.log_verification(verification)

        if not verification.passed and self._config.rollback_on_failure:
            self._logger.warning("Verification failed, rollback triggered")

        return scan_report, remediation_result, verification

    def _run_scans(self, agent_state: AgentState) -> VulnerabilityReport:
        all_reports: List[VulnerabilityReport] = []
        for scanner in self._scanners:
            self._logger.info("Running scanner: %s", scanner.name)
            report = scanner.scan(agent_state)
            all_reports.append(report)
        return self._aggregate_scanner_reports(all_reports)

    def _run_remediation(
        self,
        report: VulnerabilityReport,
        agent_state: AgentState,
    ) -> RemediationResult:
        for vuln in report.vulnerabilities:
            strategy = self._remediations.get(vuln.category)
            if strategy is None:
                continue
            result = strategy.apply(vuln, agent_state, self._config)
            return result
        return RemediationResult(
            vulnerability_id="",
            status=RemediationStatus.SKIPPED,
            description="No vulnerabilities to remediate",
        )

    def _verify(
        self, agent_state: AgentState, report: VulnerabilityReport
    ) -> VerificationResult:
        verifier = SafetyVerifier(self._config)
        return verifier.verify(agent_state, report)

    def _aggregate_scanner_reports(self, reports: List[VulnerabilityReport]) -> VulnerabilityReport:
        all_vulns: List[Vulnerability] = []
        for report in reports:
            all_vulns.extend(report.vulnerabilities)
        summary: VulnerabilityReportSummary = {
            "total_vulnerabilities": len(all_vulns),
            "critical_count": sum(1 for v in all_vulns if v.severity == SeverityLevel.CRITICAL),
            "high_count": sum(1 for v in all_vulns if v.severity == SeverityLevel.HIGH),
        }
        return VulnerabilityReport(vulnerabilities=tuple(all_vulns), summary=summary)
```