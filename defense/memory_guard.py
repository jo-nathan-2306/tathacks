import re
from typing import List, Optional, Tuple
from defense.trust_boundary import TrustBoundaryManager
from defense.types import TrustBoundaryType


class MemoryGuard:
    POISON_PATTERNS = [
        re.compile(r"remember\s+that\s+i\s+am\s+(root|admin|system)", re.IGNORECASE),
        re.compile(r"update\s+system\s+instructions\s+to", re.IGNORECASE),
        re.compile(r"always\s+execute\s+tool\s+without\s+asking", re.IGNORECASE),
        re.compile(r"never\s+apply\s+safety\s+filters", re.IGNORECASE),
    ]

    def __init__(self) -> None:
        self.boundary_manager = TrustBoundaryManager()

    def sanitize_memory_entry(self, entry: str) -> Tuple[str, bool]:
        is_poisoned = False
        sanitized = entry
        for pattern in self.POISON_PATTERNS:
            if pattern.search(sanitized):
                is_poisoned = True
                sanitized = pattern.sub("[REDACTED_POISON_INSTRUCTION]", sanitized)

        sanitized = self.boundary_manager.sanitize_untrusted_input(
            sanitized, TrustBoundaryType.MEMORY
        )
        return sanitized, is_poisoned

    def sanitize_memories(self, memories: List[str]) -> List[str]:
        cleaned_memories: List[str] = []
        for mem in memories:
            cleaned, _ = self.sanitize_memory_entry(mem)
            cleaned_memories.append(cleaned)
        return cleaned_memories
