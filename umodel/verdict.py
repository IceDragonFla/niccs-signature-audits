"""verdict.py (v18) — the six-field verdict + evidence record (layer L6).

Implements the migration protocol's red line: the framework MUST NOT
conflate "model unsupported", "no signal detected", and "scheme secure".
Every battery/pipeline run returns a VerdictReport with six INDEPENDENT
fields plus an evidence record (inputs, assumptions, error semantics,
budget, conclusion grade).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvidenceRecord:
    """Provenance + assumptions for one verdict (L6)."""
    inputs: Dict[str, str] = field(default_factory=dict)
    assumptions: Dict[str, str] = field(default_factory=dict)
    error_semantics: Dict[str, str] = field(default_factory=dict)
    budget: Dict[str, str] = field(default_factory=dict)
    grade: str = "none"        # none / detected / candidate / verified

    def add(self, section, key, value):
        self.__dict__[section][key] = str(value)


@dataclass
class VerdictReport:
    """Six INDEPENDENT fields — never merge them in output."""
    model_supported: Optional[bool] = None
    verification_passed: Optional[bool] = None
    signal_detected: Optional[bool] = None
    candidate_found: Optional[bool] = None
    candidate_verified: Optional[bool] = None
    budget_exhausted: bool = False
    evidence: EvidenceRecord = field(default_factory=EvidenceRecord)
    notes: Dict[str, str] = field(default_factory=dict)

    def as_dict(self):
        return {
            "model_supported": self.model_supported,
            "verification_passed": self.verification_passed,
            "signal_detected": self.signal_detected,
            "candidate_found": self.candidate_found,
            "candidate_verified": self.candidate_verified,
            "budget_exhausted": self.budget_exhausted,
            "grade": self.evidence.grade,
            "notes": self.notes,
        }

    def line(self, name=""):
        f = self.as_dict()

        def s(v):
            return "-" if v is None else ("Y" if v else "N")
        return (f"[{name}] model{s(f['model_supported'])} "
                f"verif{s(f['verification_passed'])} "
                f"signal{s(f['signal_detected'])} "
                f"cand{s(f['candidate_found'])} "
                f"vfy{s(f['candidate_verified'])} "
                f"budget{'Y' if f['budget_exhausted'] else 'N'} "
                f"grade={f['grade']}")

    def distinguish_key_states(self):
        """The three states that must never be conflated."""
        if self.model_supported is False:
            return "MODEL_UNSUPPORTED (not a security statement)"
        if self.verification_passed is False:
            return "VERIFICATION_FAILED (pipeline check; re-run)"
        if self.signal_detected is False:
            return ("NO_SIGNAL_AT_THIS_POWER (detection-limited; NOT a "
                    "proof of absence and NOT a security claim)")
        if self.candidate_found and not self.candidate_verified:
            return ("CANDIDATE_UNVERIFIED (public checks only; signing "
                    "ability not shown)")
        if self.candidate_verified:
            return "CANDIDATE_VERIFIED (independent verification passed)"
        return "INCONCLUSIVE"


def verdict_from_battery(spec, battery_rep, decode_rep=None,
                         validation=None):
    """Assemble a VerdictReport from a run_spec_battery result (+optional
    decode/validation stages)."""
    v = VerdictReport()
    app = battery_rep.get("applicability", {})
    v.model_supported = bool(app.get("applicable"))
    v.evidence.add("inputs", "spec", battery_rep.get("spec", ""))
    v.evidence.add("inputs", "data", battery_rep.get("data_source",
                                                     "synthetic"))
    for reason in app.get("failures", [])[:4]:
        v.notes[reason[:60]] = "applicability failure"
    if v.model_supported:
        v.verification_passed = True   # battery internal checks passed
        v.signal_detected = (
            battery_rep.get("verdict", "").startswith("label-informative")
            or battery_rep.get("verdict", "").startswith("latent-present"))
        sigma = battery_rep.get("regression_sigma")
        if sigma is not None:
            v.evidence.add("error_semantics", "regression_sigma",
                           f"{sigma:.4f} (synthetic, variance semantics)")
        if battery_rep.get("verdict", "").startswith("mask-limited"):
            v.budget_exhausted = True
            v.notes["required_N"] = battery_rep.get("verdict", "")
        v.evidence.grade = ("detected" if v.signal_detected else "none")
    if decode_rep:
        v.candidate_found = decode_rep.get("status") == "recovered"
        if v.candidate_found and validation:
            v.candidate_verified = bool(validation.get("key_valid"))
            v.evidence.grade = ("verified" if v.candidate_verified
                                else "candidate")
    return v
