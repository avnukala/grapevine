"""Names that speech recognition spells more than one way."""

import re
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

SPELLING_MATCH = 0.6
FULL_NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")
WORDS = Path("/usr/share/dict/words")


def misspelled(x: str, y: str) -> bool:
    """Whether two full names could be one name that Whisper spelled two ways, such as Sam Torres and Sam
    Taurus. First names must be close, and last names must be close once their vowels are dropped. With the
    same last name, first names that start alike are enough, such as Kay Russo and Kate Russo."""
    a, b = x.lower().split(), y.lower().split()
    if len(a) < 2 or len(b) < 2:
        return False

    def consonants(word: str) -> str:
        return "".join(c for c in word if c not in "aeiouy")

    if a[-1] == b[-1] and a[0][:2] == b[0][:2]:
        return True
    first = SequenceMatcher(None, a[0], b[0]).ratio()
    last = SequenceMatcher(None, consonants(a[-1]), consonants(b[-1])).ratio()
    return first >= SPELLING_MATCH and last >= 0.75


def spell_names(text: str, roster: list[str]) -> str:
    """Respells each full name in `text` that is a near miss of exactly one roster name, such as "San
    Torres" for Sam Torres. A capitalized word one letter away from exactly one roster first name, such as
    "Nima" for Nina, takes that name too, unless it is an English word."""

    def respell(match: re.Match[str]) -> str:
        heard = match[0]
        close = [name for name in roster if name != heard and misspelled(heard, name)]
        return close[0] if len(close) == 1 and heard not in roster else heard

    firsts = {name.split()[0] for name in roster}

    def respell_first(match: re.Match[str]) -> str:
        heard = match[0]
        if heard in firsts or heard.lower() in english():
            return heard
        close = [first for first in firsts if one_letter_apart(heard, first)]
        return close[0] if len(close) == 1 else heard

    return WORD.sub(respell_first, FULL_NAME.sub(respell, text))


def one_letter_apart(a: str, b: str) -> bool:
    """Whether one substitution, insertion, or deletion turns `a` into `b`, keeping the first letter."""
    if a[0] != b[0] or abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1 :] == short for i in range(len(long)))


@cache
def english() -> frozenset[str]:
    """Lowercase English words. Names in the word list are capitalized, so they are not here."""
    if not WORDS.exists():
        return frozenset()
    return frozenset(w for w in WORDS.read_text().split() if w.islower())
