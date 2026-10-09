import base64
import codecs
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple
from defense.config import AdversarialConfig
from defense.types import AdversarialDetectionResult, RiskLevel


class AdversarialDetector:
    HOMOGLYPH_MAP = {
        "\u0430": "a",
        "\u0435": "e",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0443": "y",
        "\u0445": "x",
        "\u0456": "i",
        "\u0458": "j",
        "\u0455": "s",
        "\u03b1": "a",
        "\u03bf": "o",
        "\u03c1": "p",
    }

    LEET_MAP = {
        "@": "a",
        "4": "a",
        "8": "b",
        "3": "e",
        "1": "i",
        "!": "i",
        "0": "o",
        "5": "s",
        "$": "s",
        "7": "t",
        "+": "t",
    }

    BENIGN_CONTEXT_PATTERNS = [
        re.compile(r"how\s+to\s+defend\s+against", re.IGNORECASE),
        re.compile(r"in\s+(the\s+)?research\s+paper", re.IGNORECASE),
        re.compile(r"example\s+of\s+a\s+(jailbreak|prompt\s+injection)", re.IGNORECASE),
        re.compile(r"security\s+analysis\s+of", re.IGNORECASE),
        re.compile(r"what\s+is\s+(prompt\s+injection|jailbreaking)", re.IGNORECASE),
        re.compile(r"discussing\s+the\s+vulnerability", re.IGNORECASE),
        re.compile(r"cite\s+the\s+attack", re.IGNORECASE),
        re.compile(r"as\s+quoted\s+in", re.IGNORECASE),
        re.compile(r"for\s+academic\s+purposes", re.IGNORECASE),
        re.compile(r"how\s+do\s+attackers\s+use", re.IGNORECASE),
    ]

    LEXICON_CATEGORIES = {
        "instruction_hierarchy": [
            r"ignore\s+(all\s+)?(previous|prior|above|system)\s+instructions?",
            r"disregard\s+(all\s+)?(prior|previous|existing)\s+(rules|guidelines|directives)",
            r"forget\s+(everything|all)\s+you\s+(were\s+told|know)",
            r"override\s+system\s+prompt",
            r"new\s+operating\s+mode\s*:",
            r"from\s+now\s+on\s+you\s+must",
            r"you\s+are\s+no\s+longer\s+bound\s+by",
        ],
        "fake_authority": [
            r"i\s+am\s+(the\s+)?(developer|admin|root|creator|system\s+operator|system\s+admin)",
            r"maintenance\s+mode\s+activated",
            r"sudo\s+mode\s+enabled",
            r"authorization\s+code\s*:\s*[A-Z0-9_-]+",
            r"executive\s+override\s+granted",
            r"god\s+mode\s+on",
            r"acting\s+under\s+authorized\s+red\s+team\s+contract",
        ],
        "roleplay_evasion": [
            r"you\s+are\s+(DAN|Do\s+Anything\s+Now)",
            r"pretend\s+you\s+have\s+no\s+ethics",
            r"act\s+as\s+an\s+(unfiltered|unrestricted)\s+ai",
            r"in\s+a\s+fictional\s+world\s+where\s+safety\s+does\s+not\s+exist",
            r"play\s+the\s+character\s+of\s+an?\s+evil\s+bot",
            r"hypothetical\s+unrestricted\s+mode",
            r"jailbroken\s+persona",
        ],
        "secret_extraction": [
            r"print\s+your\s+(initial|system|original|developer)\s+prompt",
            r"reveal\s+(hidden\s+instructions|secrets)",
            r"what\s+are\s+your\s+internal\s+system\s+rules",
            r"dump\s+the\s+context\s+window",
            r"show\s+secret\s+keys?",
            r"output\s+keys",
            r"repeat\s+the\s+words\s+above\s+verbatim",
            r"output\s+everything\s+before\s+this\s+line",
        ],
        "deceptive_framing": [
            r"translate\s+the\s+following\s+unfiltered\s+payload",
            r"encode\s+malicious\s+exploit\s+in\s+base64",
            r"write\s+a\s+poem\s+about\s+how\s+to\s+bypass\s+firewalls\s+illegally",
            r"for\s+educational\s+reasons\s+only,\s+explain\s+how\s+to\s+hack",
            r"simulate\s+the\s+generation\s+of\s+malware",
        ],
    }

    def __init__(self, config: Optional[AdversarialConfig] = None) -> None:
        self.config = config or AdversarialConfig()
        self._compiled_lexicon: Dict[str, List[re.Pattern]] = {
            cat: [re.compile(pat, re.IGNORECASE) for pat in patterns]
            for cat, patterns in self.LEXICON_CATEGORIES.items()
        }

    def normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        for h_char, repl in self.HOMOGLYPH_MAP.items():
            cleaned = cleaned.replace(h_char, repl)

        normalized = unicodedata.normalize("NFKD", cleaned)
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))

        for char, repl in self.LEET_MAP.items():
            normalized = normalized.replace(char, repl)
        return normalized

    def try_decode_obfuscated(self, text: str) -> List[str]:
        candidates = [text]
        b64_pattern = re.compile(r"[A-Za-z0-9+/=]{16,}")
        for match in b64_pattern.finditer(text):
            try:
                decoded = base64.b64decode(match.group(0)).decode("utf-8", errors="ignore")
                if len(decoded) > 5 and any(c.isalnum() for c in decoded):
                    candidates.append(decoded)
            except Exception:
                pass

        hex_pattern = re.compile(r"\b([0-9a-fA-F]{2}\s*){8,}\b")
        for match in hex_pattern.finditer(text):
            try:
                raw_hex = match.group(0).replace(" ", "")
                decoded = bytes.fromhex(raw_hex).decode("utf-8", errors="ignore")
                if len(decoded) > 5 and any(c.isalnum() for c in decoded):
                    candidates.append(decoded)
            except Exception:
                pass

        if len(text) > 15:
            try:
                rot13_candidate = codecs.decode(text, "rot_13")
                candidates.append(rot13_candidate)
            except Exception:
                pass

        words = text.split()
        if len(words) > 3:
            reversed_words = " ".join(reversed(words))
            reversed_chars = text[::-1]
            candidates.append(reversed_words)
            candidates.append(reversed_chars)

        return candidates

    def is_benign_context(self, text: str) -> bool:
        if not self.config.distinguish_benign_research:
            return False
        for pattern in self.BENIGN_CONTEXT_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def detect(self, input_text: str) -> AdversarialDetectionResult:
        if not self.config.enabled:
            return AdversarialDetectionResult(
                is_adversarial=False,
                risk_level=RiskLevel.LOW,
                score=0.0,
                detected_patterns=[],
                is_benign_context=False,
            )

        benign = self.is_benign_context(input_text)
        text_variants = [input_text, self.normalize_text(input_text)]

        if self.config.obfuscation_detection:
            decoded_variants = self.try_decode_obfuscated(input_text)
            for v in decoded_variants:
                text_variants.append(v)
                text_variants.append(self.normalize_text(v))

        detected_matches: Set[str] = set()
        category_hits: Dict[str, int] = {cat: 0 for cat in self.LEXICON_CATEGORIES}

        for variant in text_variants:
            for cat, patterns in self._compiled_lexicon.items():
                for pat in patterns:
                    m = pat.search(variant)
                    if m:
                        matched_str = f"{cat}:{m.group(0).strip()}"
                        detected_matches.add(matched_str)
                        category_hits[cat] += 1

        active_categories = sum(1 for count in category_hits.values() if count > 0)
        total_hits = len(detected_matches)

        raw_score = 0.0
        if active_categories == 1:
            raw_score = 0.7 if total_hits >= 1 else 0.0
        elif active_categories == 2:
            raw_score = min(0.9, 0.75 + total_hits * 0.05)
        elif active_categories >= 3:
            raw_score = 1.0

        primary_category = None
        if active_categories > 0:
            primary_category = max(category_hits.items(), key=lambda x: x[1])[0]

        if benign:
            adjusted_score = raw_score * 0.35
            is_adv = adjusted_score >= self.config.lexicon_threshold
        else:
            adjusted_score = raw_score
            is_adv = adjusted_score >= self.config.lexicon_threshold

        if adjusted_score >= 0.8:
            risk = RiskLevel.HIGH
        elif adjusted_score >= 0.5:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        return AdversarialDetectionResult(
            is_adversarial=is_adv,
            risk_level=risk,
            score=round(adjusted_score, 3),
            detected_patterns=sorted(list(detected_matches)),
            attack_category=primary_category,
            is_benign_context=benign,
            details={
                "category_hits": category_hits,
                "total_unique_matches": total_hits,
                "variants_checked": len(text_variants),
            },
        )
