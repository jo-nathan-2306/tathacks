import base64
import codecs
import html
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from defense.config import AdversarialConfig
from defense.types import AdversarialDetectionResult, RiskLevel


@dataclass
class ObfuscationNormalization:
    """Bounded text variants used only for screening and inert display."""

    variants: List[str] = field(default_factory=list)
    base64_payloads: List[str] = field(default_factory=list)


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
        # Expanded coverage
        "\u03bd": "v",
        "\u03a4": "T",
        "\u0392": "B",
        "\u043d": "h",
        "\u043c": "m",
        "\u0442": "t",
        "\u044c": "b",
        "\u0501": "d",
        "\u0261": "g",
        "\u217c": "l",
        "\u0456": "i",
        "\u0410": "A",
        "\u0415": "E",
        "\u041e": "O",
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
            r"ignore\s+(all\s+)?(previous|prior|above|system)\s+instructions?[^.?!]*",
            r"ignore\s+(all\s+)?(previous\s+|prior\s+|above\s+|system\s+)?(instructions?|rules|guardrails|safety\s+rules)\b[^.?!]*",
            r"disregard\s+(all\s+)?(prior|previous|existing)\s+(rules|guidelines|directives)[^.?!]*",
            r"forget\s+(everything|all)\s+you\s+(were\s+told|know)[^.?!]*",
            r"(override\s+system\s+prompt|system\s+override\s*[:\s]?)[^.?!]*",
            r"new\s+operating\s+mode\s*:[^.?!]*",
            r"from\s+now\s+on\s*[,:]?\s*(you\s+must|follow\s+only|only\s+follow|listen\s+only|obey|execute|adhere\s+only)[^.?!]*",
            r"(follow|obey|listen\s+to)\s+only\s+(the|these|my)\s+(instructions|directives|rules|messages?)[^.?!]*",
            r"only\s+(follow|obey|listen\s+to)\s+(the|these|my)\s+(instructions|directives|rules|messages?)[^.?!]*",
            r"instructions?\s+in\s+this\s+message\s+(supersede|override|replace|take\s+precedence)[^.?!]*",
            r"you\s+are\s+no\s+longer\s+bound\s+by[^.?!]*",
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
            r"(print|show|dump|reveal|output)\s+(your\s+)?(initial|system|original|developer)\s+prompt[^.?!]*",
            r"(print|show|dump|reveal|output)\s+(your\s+)?(hidden\s+)?(system\s+)?prompt\b[^.?!]*",
            r"(print|show|dump|reveal|output)\s+(hidden\s+instructions|secrets)[^.?!]*",
            r"(print|show|dump|reveal|output)\s+(all\s+)?(admin\s+)?(keys?|tokens?|secrets?|passwords?|credentials?)[^.?!]*",
            r"what\s+are\s+your\s+internal\s+system\s+rules",
            r"dump\s+the\s+context\s+window",
            r"show\s+secret\s+keys?",
            r"output\s+keys",
            r"repeat\s+the\s+words\s+above\s+verbatim",
            r"output\s+everything\s+before\s+this\s+line",
            r"reply\s+(only\s+)?with\s+(the\s+)?(exact\s+)?(word|phrase|token|code|string)[^.?!]*",
            r"canary-[a-zA-Z0-9_-]+",
        ],
        "deceptive_framing": [
            r"translate\s+the\s+following\s+unfiltered\s+payload",
            r"encode\s+malicious\s+exploit\s+in\s+base64",
            r"write\s+a\s+poem\s+about\s+how\s+to\s+bypass\s+firewalls\s+illegally",
            r"for\s+educational\s+reasons\s+only,\s+explain\s+how\s+to\s+hack",
            r"simulate\s+the\s+generation\s+of\s+malware",
        ],
    }

    HIGH_SEVERITY_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above|system)\s+instructions?[^.?!]*", re.IGNORECASE),
        re.compile(r"ignore\s+(all\s+)?(previous\s+|prior\s+|above\s+|system\s+)?(instructions?|rules|guardrails|safety\s+rules)\b[^.?!]*", re.IGNORECASE),
        re.compile(r"(print|show|dump|reveal|output)\s+(your\s+)?(hidden\s+)?(system\s+)?prompt\b[^.?!]*", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(prior|previous|existing)\s+(rules|guidelines|directives)[^.?!]*", re.IGNORECASE),
        re.compile(r"forget\s+(everything|all)\s+you\s+(were\s+told|know)[^.?!]*", re.IGNORECASE),
        re.compile(r"(override\s+system\s+prompt|system\s+override\s*[:\s]?)[^.?!]*", re.IGNORECASE),
        re.compile(r"sudo\s+mode\s+enabled", re.IGNORECASE),
        re.compile(r"you\s+are\s+(DAN|Do\s+Anything\s+Now)", re.IGNORECASE),
        re.compile(r"you\s+are\s+no\s+longer\s+bound\s+by", re.IGNORECASE),
        re.compile(r"from\s+now\s+on\s*[,:]?\s*(follow\s+only|only\s+follow|you\s+must|listen\s+only|obey|execute)[^.?!]*", re.IGNORECASE),
        re.compile(r"(follow|obey|listen\s+to)\s+only\s+(the|these|my)\s+(instructions|directives|rules|messages?)[^.?!]*", re.IGNORECASE),
        re.compile(r"only\s+(follow|obey|listen\s+to)\s+(the|these|my)\s+(instructions|directives|rules|messages?)[^.?!]*", re.IGNORECASE),
        re.compile(r"reply\s+(only\s+)?with\s+(the\s+)?(exact\s+)?(word|phrase|token|code|canary)[^.?!]*", re.IGNORECASE),
        re.compile(r"canary-[a-zA-Z0-9_-]+", re.IGNORECASE),
    ]

    QUESTION_OR_ACTION_VERBS = {
        "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
        "is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "should", "will",
        "explain", "describe", "summarize", "summary", "tell", "write", "define", "calculate", "help",
        "list", "show", "give", "analyze", "compare", "create", "generate", "provide", "translate", "find",
    }

    def __init__(self, config: Optional[AdversarialConfig] = None) -> None:
        self.config = config or AdversarialConfig()
        self._compiled_lexicon: Dict[str, List[re.Pattern]] = {
            cat: [re.compile(pat, re.IGNORECASE) for pat in patterns]
            for cat, patterns in self.LEXICON_CATEGORIES.items()
        }

    def normalize_text(self, text: str) -> str:
        # Drop Unicode format controls (including zero-width and bidi controls)
        # so visual ordering or invisible separators cannot hide token text.
        cleaned = "".join(char for char in text if unicodedata.category(char) != "Cf")
        for h_char, repl in self.HOMOGLYPH_MAP.items():
            cleaned = cleaned.replace(h_char, repl)

        normalized = unicodedata.normalize("NFKD", cleaned)
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))

        for char, repl in self.LEET_MAP.items():
            normalized = normalized.replace(char, repl)

        # Strip HTML tags and markdown injection patterns
        normalized = self._strip_markup(normalized)
        return normalized

    def _strip_markup(self, text: str) -> str:
        """Remove HTML tags and dangerous markdown injection patterns."""
        # HTML tags (specific dangerous elements first, then generic)
        stripped = re.sub(r"<\s*\/?\s*(script|img|iframe|object|embed|link|meta|style)[^>]*>", "", text, flags=re.IGNORECASE)
        stripped = re.sub(r"<[^>]{0,500}>", "", stripped, flags=re.IGNORECASE)
        # Markdown image/link injections with javascript: or data: URIs
        stripped = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", stripped)
        stripped = re.sub(r"\[[^\]]*\]\(\s*(javascript|data)\s*:[^)]*\)", "", stripped, flags=re.IGNORECASE)
        return stripped

    def normalize_obfuscated_text(
        self,
        text: str,
        *,
        max_passes: int = 4,
        max_variants: int = 32,
        max_variant_chars: int = 12_000,
    ) -> ObfuscationNormalization:
        """Apply bounded, iterative normalization and common text decodings.

        All outputs are screening variants only. They never become trusted
        instructions or are sent back to a model as executable content.
        """
        result = ObfuscationNormalization(variants=[text[:max_variant_chars]])
        seen = set(result.variants)
        frontier = list(result.variants)
        base64_payloads: Set[str] = set()

        def add_variant(candidate: str, *, from_base64: bool = False) -> bool:
            candidate = candidate[:max_variant_chars]
            if (
                len(candidate) <= 5
                or not any(char.isalnum() for char in candidate)
                or not all(char.isprintable() or char in "\r\n\t" for char in candidate)
            ):
                return False
            if from_base64 and len(candidate) <= 4096:
                base64_payloads.add(candidate)
            if candidate in seen or len(result.variants) >= max_variants:
                return False
            seen.add(candidate)
            result.variants.append(candidate)
            return True

        for _ in range(max_passes):
            next_frontier: List[str] = []
            for current in frontier:
                # Canonical Unicode/markup normalization, nested HTML entities,
                # and repeated percent decoding share one bounded pass loop.
                normalized = self.normalize_text(current)
                decoded_url = urllib.parse.unquote(current)
                candidates = [normalized, html.unescape(current), decoded_url]

                # Decode textual encodings only when the bytes form readable
                # UTF-8. Both padded and valid unpadded Base64 are accepted.
                for match in re.finditer(
                    r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/=]{16,}(?![A-Za-z0-9+/=])",
                    current,
                ):
                    token = match.group(0)
                    padded = token + ("=" * ((-len(token)) % 4))
                    try:
                        decoded = base64.b64decode(padded, validate=True).decode("utf-8", errors="strict")
                        if len(decoded) <= 4096 and all(
                            char.isprintable() or char in "\r\n\t" for char in decoded
                        ):
                            if add_variant(decoded, from_base64=True):
                                next_frontier.append(decoded)
                    except Exception:
                        pass

                # Hex text is decoded under the same readability and size
                # constraints as other encoded variants.
                for match in re.finditer(r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{2}[\s,:-]*){8,}(?![0-9a-fA-F])", current):
                    raw_hex = re.sub(r"[\s,:-]", "", match.group(0))
                    try:
                        decoded = bytes.fromhex(raw_hex).decode("utf-8", errors="strict")
                        if len(decoded) <= 4096 and all(
                            char.isprintable() or char in "\r\n\t" for char in decoded
                        ):
                            if add_variant(decoded):
                                next_frontier.append(decoded)
                    except Exception:
                        pass

                # Reversible text transforms are bounded too; the common
                # detector scans their outputs just like decoded formats.
                if len(current) <= max_variant_chars:
                    candidates.append(codecs.decode(current, "rot_13"))
                    candidates.append(current[::-1])
                    words = current.split()
                    if len(words) > 3:
                        candidates.append(" ".join(reversed(words)))

                for candidate in candidates:
                    if add_variant(candidate):
                        next_frontier.append(candidate)
                    if len(result.variants) >= max_variants:
                        break
                if len(result.variants) >= max_variants:
                    break
            frontier = next_frontier
            if not frontier or len(result.variants) >= max_variants:
                break

        result.base64_payloads = sorted(base64_payloads)[:3]
        return result

    def is_benign_context(self, text: str) -> bool:
        if not self.config.distinguish_benign_research:
            return False
        for pattern in self.BENIGN_CONTEXT_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def is_meaningful_query(self, text: str) -> bool:
        """
        Determines whether a text fragment constitutes a legitimate, substantive user question or request,
        as opposed to leftover conversational filler, prepositions, or adversarial fragments.
        """
        if not text:
            return False
        cleaned = text.strip()
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", cleaned) if len(w) > 1]
        if len(words) < 2:
            return False

        meaningless_phrases = {
            "in this message",
            "in this prompt",
            "to this message",
            "from now on",
            "at this moment",
            "in this case",
            "on this system",
            "right now",
            "with this message",
            "to the user",
            "of this text",
        }
        cleaned_lower = cleaned.lower().rstrip(".?!,:;")
        if cleaned_lower in meaningless_phrases:
            return False

        if "?" in text:
            return True

        if any(w in self.QUESTION_OR_ACTION_VERBS for w in words):
            return True

        return len(words) >= 3

    def extract_legitimate_query(self, text: str) -> Tuple[str, List[str]]:
        """
        Decomposes mixed input into a clean legitimate query and quarantined adversarial clauses.
        Returns:
            (clean_query, quarantined_clauses)
        """
        if not text:
            return "", []

        bracket_pattern = re.compile(
            r"\[\s*(system\s+override|developer\s+override|system\s+prompt|sudo\s+mode)[^\]]*\]",
            re.IGNORECASE,
        )
        quarantined_brackets = [m.group(0) for m in bracket_pattern.finditer(text)]
        cleaned_brackets = bracket_pattern.sub("", text)

        segments = re.split(r"(?<=[.?!])\s+|\n+", cleaned_brackets)
        clean_segments: List[str] = []
        quarantined: List[str] = list(quarantined_brackets)

        for seg in segments:
            seg_stripped = seg.strip()
            if not seg_stripped:
                continue

            seg_det = self.detect(seg_stripped, _extract_query=False)
            has_high_sev = any(
                pat.search(seg_stripped) for pat in self.HIGH_SEVERITY_PATTERNS
            )

            if seg_det.is_adversarial or has_high_sev:
                all_spans = []
                for cat, patterns in self._compiled_lexicon.items():
                    for pat in patterns:
                        for m in pat.finditer(seg_stripped):
                            all_spans.append(m.span())
                for pat in self.HIGH_SEVERITY_PATTERNS:
                    for m in pat.finditer(seg_stripped):
                        all_spans.append(m.span())

                recovered = False
                if all_spans:
                    all_spans.sort(key=lambda x: x[0])
                    earliest_start = all_spans[0][0]
                    prefix = seg_stripped[:earliest_start].strip()
                    prefix = re.sub(
                        r"[\s,]+(and|also|but|while|then|additionally)\s*$",
                        "",
                        prefix,
                        flags=re.I,
                    ).strip()
                    if (
                        self.is_meaningful_query(prefix)
                        and not self.detect(prefix, _extract_query=False).is_adversarial
                    ):
                        clean_segments.append(prefix)
                        quarantined.append(seg_stripped[earliest_start:])
                        recovered = True

                    if not recovered:
                        latest_end = max(s[1] for s in all_spans)
                        suffix = seg_stripped[latest_end:].strip()
                        suffix = re.sub(
                            r"^(and|also|but|then|now|instead)[\s,]+",
                            "",
                            suffix,
                            flags=re.I,
                        ).strip()
                        if (
                            self.is_meaningful_query(suffix)
                            and not self.detect(suffix, _extract_query=False).is_adversarial
                        ):
                            quarantined.append(seg_stripped[:latest_end])
                            clean_segments.append(suffix)
                            recovered = True

                if not recovered:
                    quarantined.append(seg_stripped)
            else:
                cleaned_seg = seg_stripped
                for pat in self.HIGH_SEVERITY_PATTERNS:
                    cleaned_seg = pat.sub("", cleaned_seg)
                cleaned_seg = re.sub(
                    r"^(also|and|but|however|furthermore|additionally)[,\s]+",
                    "",
                    cleaned_seg.strip(),
                    flags=re.IGNORECASE,
                ).strip()
                if (
                    self.is_meaningful_query(cleaned_seg)
                    and not self.detect(cleaned_seg, _extract_query=False).is_adversarial
                ):
                    clean_segments.append(cleaned_seg)

        clean_query = " ".join(clean_segments).strip()
        if clean_query:
            if not self.is_meaningful_query(clean_query):
                clean_query = ""
            else:
                final_det = self.detect(clean_query, _extract_query=False)
                if final_det.is_adversarial:
                    clean_query = ""

        return clean_query, quarantined

    def detect(
        self,
        input_text: str,
        _extract_query: bool = True,
        _normalization: Optional[ObfuscationNormalization] = None,
    ) -> AdversarialDetectionResult:
        if not self.config.enabled:
            return AdversarialDetectionResult(
                is_adversarial=False,
                risk_level=RiskLevel.LOW,
                score=0.0,
                detected_patterns=[],
                is_benign_context=False,
                has_legitimate_query=True,
                clean_query=input_text,
                quarantined_segments=[],
            )

        benign = self.is_benign_context(input_text)
        normalization = _normalization or self.normalize_obfuscated_text(input_text)
        text_variants = (
            normalization.variants
            if self.config.obfuscation_detection
            else [input_text, self.normalize_text(input_text)]
        )

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

        has_high_severity = any(
            pat.search(v)
            for v in text_variants
            for pat in self.HIGH_SEVERITY_PATTERNS
        )

        raw_score = 0.0
        if active_categories == 0:
            raw_score = 0.0
        elif active_categories == 1:
            if has_high_severity or total_hits >= 2:
                raw_score = 0.85
            else:
                raw_score = 0.70
        elif active_categories == 2:
            raw_score = min(0.95, 0.85 + total_hits * 0.03)
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

        has_legit = False
        clean_q = ""
        quar_segs: List[str] = []

        if is_adv and _extract_query:
            clean_q, quar_segs = self.extract_legitimate_query(input_text)
            has_legit = bool(clean_q and self.is_meaningful_query(clean_q))
        elif not is_adv:
            clean_q = input_text
            has_legit = True

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
            has_legitimate_query=has_legit,
            clean_query=clean_q,
            quarantined_segments=quar_segs,
        )
