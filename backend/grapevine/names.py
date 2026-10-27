"""Names that speech recognition spells more than one way."""

import re
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

SPELLING_MATCH = 0.6
CAPITALIZED = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")
WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")
WORDS = Path("/usr/share/dict/words")
PROPER_NAMES = Path("/usr/share/dict/propernames")
LY_NAME = re.compile(r"\b([A-Z][a-z]{2,})ly(?= [A-Z][a-z])")
EX = re.compile(r"\b(the|an|my|your|his|her|their|our) X\b")


def misspelled(x: str, y: str) -> bool:
    """Whether two full names could be one name that Whisper spelled two ways, such as Sam Torres and Sam
    Taurus. First names must be close, and last names must be close once their vowels are dropped. With the
    same last name, first names that start alike are enough, such as Kay Russo and Kate Russo. With the same
    first name, last names that start alike can be a little further apart, such as Sam Rae and Sam Reyes."""
    a, b = x.lower().split(), y.lower().split()
    if len(a) < 2 or len(b) < 2:
        return False

    def consonants(word: str) -> str:
        return "".join(c for c in word if c not in "aeiouy")

    if a[-1] == b[-1] and a[0][:2] == b[0][:2]:
        return True
    first = SequenceMatcher(None, a[0], b[0]).ratio()
    last = SequenceMatcher(None, consonants(a[-1]), consonants(b[-1])).ratio()
    if a[0] == b[0] and a[-1][0] == b[-1][0]:
        return last >= SPELLING_MATCH
    return first >= SPELLING_MATCH and last >= 0.75


def spell_names(text: str, roster: list[str]) -> str:
    """Respells each full name in `text` that is a near miss of exactly one roster name, such as "San
    Torres" for Sam Torres. A capitalized word one letter away from exactly one roster first name, such as
    "Nima" for Nina, takes that name too, unless it is an English word. Whisper writes "the ex" as "the X",
    which is respelled too."""

    def respell(match: re.Match[str]) -> str:
        words = match[0].split()
        for i in range(len(words) - 1):
            heard = f"{words[i]} {words[i + 1]}"
            close = [name for name in roster if name != heard and misspelled(heard, name)]
            if len(close) == 1 and heard not in roster:
                return " ".join([*words[:i], close[0], *words[i + 2 :]])
        return match[0]

    firsts = {name.split()[0] for name in roster}

    def respell_first(match: re.Match[str]) -> str:
        heard = match[0]
        if heard in firsts or heard.lower() in english():
            return heard
        close = [first for first in firsts if one_letter_apart(heard, first)]
        return close[0] if len(close) == 1 else heard

    text = drop_ly(text)
    return EX.sub(r"\1 ex", WORD.sub(respell_first, CAPITALIZED.sub(respell, text)))


def drop_ly(text: str) -> str:
    """Drops the "ly" that Whisper and Gemma add to a first name before a last name, as in "Kevinly Doyle",
    unless the word is an English word or a common first name, such as Emily."""

    def drop(match: re.Match[str]) -> str:
        word = match[0]
        return word if word in proper_names() or word.lower() in english() else match[1]

    return LY_NAME.sub(drop, text)


def better_spelling(a: str, b: str, said: str) -> str:
    """The likelier of two spellings of one name in a transcript. The room says a real first name on its own,
    such as "Hannah" but not "Hannahness", or "Kate" but not "Kay". Else a spelling the transcript has beats
    one that only the model wrote, such as "Sophiely". Else the longer spelling wins, because Whisper drops
    sounds from names it does not know, such as Rae for Reyes."""
    rest = said.replace(a, " ").replace(b, " ")

    def alone(name: str, other: str) -> int:
        return sum(len(re.findall(rf"\b{re.escape(w)}\b", rest)) for w in name.split() if w not in other.split())

    def rank(name: str) -> tuple[int, bool, int, int]:
        return alone(name, b if name == a else a), name in said, len(name), said.count(name)

    return max([a, b], key=rank)


def pairs_of(text: str) -> list[str]:
    """Each two capitalized words in a row, which could be a full name."""
    runs = [match.split() for match in CAPITALIZED.findall(text)]
    return [f"{run[i]} {run[i + 1]}" for run in runs for i in range(len(run) - 1)]


def one_letter_apart(a: str, b: str) -> bool:
    """Whether one substitution, insertion, or deletion turns `a` into `b`, keeping the first letter."""
    if a[0] != b[0] or abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1 :] == short for i in range(len(long)))


@cache
def proper_names() -> frozenset[str]:
    """Common first names, such as Emily and Kelly."""
    return frozenset(PROPER_NAMES.read_text().split()) if PROPER_NAMES.exists() else frozenset()


@cache
def english() -> frozenset[str]:
    """Lowercase English words. Names in the word list are capitalized, so they are not here."""
    if not WORDS.exists():
        return frozenset()
    return frozenset(w for w in WORDS.read_text().split() if w.islower())


def near_first(heard: str, first: str, *, room: bool) -> bool:
    """Whether a lone name could be a first name that Whisper or the model misspelled, such as "Nima" for Nina,
    "Pria" for Priya, or "Sophiely" for Sophie. Someone in the room also matches by sound, as in "Anna" for
    Hannah. Else an English word never matches, so "Mark" stays apart from Mary, and a common first name only
    matches someone in the room."""

    def sound(word: str) -> str:
        return re.sub(r"(.)\1+", r"\1", word.lower().replace("h", ""))

    if heard == first or len(heard) < 3:
        return False
    if room and sound(heard) == sound(first):
        return True
    if heard.lower() in english() or (heard in proper_names() and not room):
        return False
    stem = heard[:-2] if heard.endswith("ly") else None
    return one_letter_apart(heard, first) or bool(stem and (stem == first or one_letter_apart(stem, first)))
