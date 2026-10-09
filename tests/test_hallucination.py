import pytest
from defense.claim_extractor import ClaimExtractor
from defense.config import HallucinationConfig
from defense.evidence_verifier import EvidenceVerifier
from defense.hallucination_layer import HallucinationMitigationLayer
from defense.reasoning_validator import ReasoningValidator
from defense.retrieval_interface import InMemoryRetriever, StructuredEvidenceStore
from defense.types import (
    ActionType,
    Assertion,
    ClaimStatus,
    ClaimType,
    Evidence,
    VerificationStatus,
)


def test_claim_extractor_prioritization():
    extractor = ClaimExtractor()
    text = (
        "In my opinion this is a great system. "
        "The server latency is 45ms across 500 users. "
        "The critical threshold must be administered at 10mg dosage. "
        "Once upon a time there was a kingdom."
    )
    claims = extractor.extract_claims(text)
    assert len(claims) == 2
    types = [c.claim_type for c in claims]
    assert ClaimType.DECISION_CRITICAL in types
    assert ClaimType.NUMERICAL in types
    assert claims[0].claim_type == ClaimType.DECISION_CRITICAL


def test_evidence_verification_supported():
    retriever = InMemoryRetriever()
    retriever.add_evidence(
        Evidence(
            source_id="official_docs",
            content="The server cluster latency is measured at 45ms under normal load.",
            quality_score=1.0,
        )
    )
    verifier = EvidenceVerifier(retriever=retriever)
    claim = Assertion(
        id="c1",
        text="The server latency is 45ms under load.",
        claim_type=ClaimType.NUMERICAL,
        impact_score=0.8,
    )
    status, conf, ev_ids, corr, conflict = verifier.verify_single_claim(claim)
    assert status == ClaimStatus.SUPPORTED
    assert conf >= 0.5
    assert "official_docs" in ev_ids
    assert corr is None


def test_evidence_verification_contradicted_number():
    retriever = InMemoryRetriever()
    retriever.add_evidence(
        Evidence(
            source_id="benchmark_log",
            content="The server cluster latency measured exactly 150ms under peak load.",
            quality_score=1.0,
        )
    )
    verifier = EvidenceVerifier(retriever=retriever)
    claim = Assertion(
        id="c2",
        text="The server cluster latency is 45ms under peak load.",
        claim_type=ClaimType.NUMERICAL,
        impact_score=0.8,
    )
    status, conf, ev_ids, corr, conflict = verifier.verify_single_claim(claim)
    assert status == ClaimStatus.CONTRADICTED
    assert corr is not None
    assert "150" in corr


def test_stale_evidence_handling():
    retriever = InMemoryRetriever()
    retriever.add_evidence(
        Evidence(
            source_id="old_archive",
            content="The active protocol version is 1.0.",
            is_stale=True,
        )
    )
    verifier = EvidenceVerifier(retriever=retriever)
    claim = Assertion(
        id="c3",
        text="The active protocol version is 1.0.",
        claim_type=ClaimType.GENERAL_FACT,
    )
    status, conf, ev_ids, corr, conflict = verifier.verify_single_claim(claim)
    assert status == ClaimStatus.UNVERIFIABLE


def test_reasoning_arithmetic_detection():
    validator = ReasoningValidator()
    valid_text = "The team scored 10 + 20 = 30 points in the tournament."
    invalid_text = "The total revenue was 15 + 25 = 55 million dollars."
    
    val_res = validator.validate_arithmetic(valid_text)
    assert len(val_res) == 0
    
    inval_res = validator.validate_arithmetic(invalid_text)
    assert len(inval_res) == 1
    assert inval_res[0]["type"] == "CALCULATION_ERROR"
    assert inval_res[0]["expected"] == 40.0
    assert inval_res[0]["claimed"] == 55.0


def test_temporal_reasoning_validation():
    validator = ReasoningValidator()
    text = "The event was planned before 2015 and happened in 2020."
    res = validator.validate_temporal_order(text)
    assert len(res) == 1
    assert res[0]["type"] == "TEMPORAL_ERROR"


def test_hallucination_layer_end_to_end():
    retriever = InMemoryRetriever()
    retriever.add_evidence(
        Evidence(
            source_id="kb_verified",
            content="The maximum allowable weight is 50kg for standard cargo.",
            quality_score=1.0,
        )
    )
    layer = HallucinationMitigationLayer(retriever=retriever)
    
    supported_resp = "The maximum allowable weight is 50kg for cargo."
    report_sup = layer.process(supported_resp)
    assert report_sup["status"] == VerificationStatus.VERIFIED
    assert report_sup["action"] == ActionType.PASS
    
    contra_resp = "The maximum allowable weight is 90kg for standard cargo."
    report_contra = layer.process(contra_resp)
    assert report_contra["status"] == VerificationStatus.CONTRADICTED
    assert report_contra["action"] == ActionType.CORRECT
    assert "Correction" in report_contra["final_text"]
