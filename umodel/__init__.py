"""umodel — a transferable-analysis framework for shared-latent leakage.

Status (v3): RESEARCH FRAMEWORK with explicit, machine-checked
applicability scope; no universality beyond the declared family.

What it does:
  1. SchemeSpec describes a scheme instance declaratively: ring structure,
     mask models, channels, and the public-key relation. The same
     engines/simulator/decoders run on any spec — verified by transfer to
     a cyclic-ring, signed-challenge toy instance (demos 5-7).
  2. Engines turn public transcripts into noisy estimates of LINKED
     quantities. The composition law governs how estimates may be
     combined: channels linked by a mod-q relation cannot be fused
     linearly; fusion happens only through the constrained decoder.
  3. The decoder ladder is sigma-adaptive and self-certifying (public-
     relation razor); it is the ONLY place cross-channel fusion occurs.
  4. At ANY sample size the pipeline returns either a decoded candidate or
     a certified analog answer (interval + required-N) — data-volume
     agnostic by contract, not by assumption.
  5. check_applicability REFUSES schemes outside the implemented scope
     (spectral basis, moment-operator derivation) with explicit reasons.

Composition law (validated three times in the audit): channels estimating
quantities linked by a mod-q relation cannot be fused linearly. Sample
complexity is governed by the best single channel; other channels
contribute correctness checks and cross-validation.
"""
from .spec import (SchemeSpec, ChannelSpec, PublicKeyRelation,
                   RingStructure, MaskModel, check_applicability)
from .channels import LabelChannel, ShiftRegressionChannel, \
    QuantizedMomentChannel
from .engines import LabelEngine, Z1WlsEngine, HintOperatorEngine
from .decoder import DecoderLadder
from .diagnose import SharedLatentDiagnosis
from .pipeline import UniversalPipeline
from .assay import LatentInformationAssay
from .verdict import VerdictReport, EvidenceRecord, verdict_from_battery
from . import simulate, linalg, niccs, public_decoders

__all__ = ["SchemeSpec", "ChannelSpec", "PublicKeyRelation",
           "RingStructure", "MaskModel", "check_applicability",
           "LabelChannel", "ShiftRegressionChannel", "QuantizedMomentChannel",
           "LabelEngine", "Z1WlsEngine", "HintOperatorEngine",
           "DecoderLadder", "SharedLatentDiagnosis", "UniversalPipeline",
           "LatentInformationAssay", "VerdictReport", "EvidenceRecord",
           "verdict_from_battery", "simulate", "linalg", "niccs",
           "public_decoders"]
