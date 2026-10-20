"""Names that speech recognition spells more than one way."""

import re
from difflib import SequenceMatcher

SPELLING_MATCH = 0.6
FULL_NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")


def misspelled(x: str, y: str) -> bool:
    """Whether two full names could be one name that Whisper spelled two ways, such as Sam Torres and Sam
    Taurus. First names must be close, and last names must be close once their vowels are dropped."""
    a, b = x.lower().split(), y.lower().split()
    if len(a) < 2 or len(b) < 2:
        return False

    def consonants(word: str) -> str:
        return "".join(c for c in word if c not in "aeiouy")

    first = SequenceMatcher(None, a[0], b[0]).ratio()
    last = SequenceMatcher(None, consonants(a[-1]), consonants(b[-1])).ratio()
    return first >= SPELLING_MATCH and last >= 0.75


def spell_names(text: str, roster: list[str]) -> str:
    """Respells each full name in `text` that is a near miss of exactly one roster name, such as "San
    Torres" for Sam Torres."""

    def respell(match: re.Match[str]) -> str:
        heard = match[0]
        close = [name for name in roster if name != heard and misspelled(heard, name)]
        return close[0] if len(close) == 1 and heard not in roster else heard

    return FULL_NAME.sub(respell, text)
