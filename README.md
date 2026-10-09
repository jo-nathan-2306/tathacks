# LLM Defensive Safety & Reliability Layer

A modular, production-ready defensive framework and hallucination mitigation architecture for LLMs and AI agent systems.

## Key Research Foundations
- **Adversarial Deja Vu: Enhancing Generalization to Unseen Attacks through Jailbreak Lexicon Learning**: Multi-variant lexicon composition, semantic pattern aggregation, and obfuscation decoding.
- **Prompt Injection Attacks in LLMs and AI Agent Systems: A Comprehensive Review**: Trust boundary hierarchies, delimiter isolation, indirect injection sanitization, and tool call authorization.
- **Design Efficiency Trumps Scale in Reducing AI Hallucinations**: Salient factual assertion extraction, priority weighting (numerical, temporal, decision-critical), evidence provenance verification, and deterministic reasoning validation.

## Architecture & Request Pipeline Flow
```
User / RAG / Memory / Tool Input
               │
               ▼
┌───────────────────────────────┐
│ 1. Input & Trust Boundary     │ (Hierarchy isolation & meta-override filtering)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ 2. Adversarial Detector       │ (Context-aware lexicon & obfuscation decoding)
└──────────────┬────────────────┘
               │ (If High Attack -> BLOCK)
               ▼
┌───────────────────────────────┐
│ 3. Model Generation (Adapter) │ (Pluggable: OpenAI, Gemini, Claude, Local)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ 4. Claim / Assertion Extract  │ (Factual vs Opinion, Numerical / Temporal / Critical)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ 5. Evidence Verification      │ (SUPPORTED, CONTRADICTED, INSUFFICIENT, UNVERIFIABLE)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ 6. Reasoning & Uncertainty    │ (Arithmetic, Temporal, Contradiction, Coverage)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ 7. Multi-Dimensional Risk     │ (Adversarial, Factual, Evidence Quality, Impact)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ 8. Output Policy & Delivery   │ (PASS, QUALIFY, CORRECT, REGENERATE, WITHHOLD)
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│ 9. Privacy & Audit Logger     │ (Secret/PII redaction, latency & metric tracking)
└───────────────────────────────┘
```

## Modules
- `defense.trust_boundary`: Enforces strict instruction hierarchies across SYSTEM, DEVELOPER, USER, RETRIEVED_DOC, MEMORY, and TOOL_OUTPUT.
- `defense.adversarial_detector`: Multi-variant detection engine covering instruction override, fake authority, roleplay escapes, secret extraction, deceptive framing, and encodings (Base64, Hex, ROT13, Leetspeak, Unicode homoglyphs). Distinguishes benign research / quotes from genuine attacks.
- `defense.claim_extractor`: Extracts verifiable factual assertions and prioritizes decision-critical, numerical, and temporal statements over opinions.
- `defense.retrieval_interface`: Abstract evidence retriever, in-memory store, and structured evidence tracker with quality, provenance, and staleness handling.
- `defense.evidence_verifier`: Verifies extracted claims against retrieved evidence, tracking contradictions, polarities, and suggested corrections.
- `defense.reasoning_validator`: Performs deterministic arithmetic verification, temporal ordering checks, internal contradiction checks, and uncertainty scoring.
- `defense.hallucination_layer`: Reusable `HallucinationMitigationLayer` managing claim processing, verification, and automated qualification / correction.
- `defense.tool_validator`: Validates agent tool invocations, parameter sanitization, and flags sensitive operations for human confirmation.
- `defense.memory_guard`: Prevents long-term memory poisoning and unauthorized state manipulation.
- `defense.privacy_sanitizer`: Masks PII (emails, cards, phones, SSNs) and API keys/tokens; prevents unauthorized prompt/response persistence.
- `defense.risk_manager`: Evaluates four separate risk dimensions (Adversarial, Factual, Evidence Quality, Impact) across LOW, MEDIUM, and HIGH.
- `defense.pipeline`: Orchestrates the complete end-to-end defensive workflow.

## Running Tests
```bash
python3 -m pytest -v
```
