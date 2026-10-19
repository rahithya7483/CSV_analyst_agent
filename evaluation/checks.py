"""Answer checking: does the model's free-text answer contain the ground-truth value?"""
import math
import re
import unicodedata

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Models often write "66 333.33" with a narrow no-break space and "ORD\u20111185" with a non-breaking hyphen.
_SPECIAL_SPACES = "\u202f\u00a0\u2009\u2007"
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"
_THOUSANDS_SPACE = re.compile("(?<=\\d)[" + _SPECIAL_SPACES + "](?=\\d{3}(?!\\d))")


def normalize(text: str) -> str:
    """Make typographic variants comparable: 66<narrow space>333.33 -> 66333.33, non-breaking hyphen -> '-'."""
    text = _THOUSANDS_SPACE.sub("", text or "")
    text = unicodedata.normalize("NFKC", text)
    return text.translate({ord(c): "-" for c in _DASHES})


def extract_numbers(text: str):
    numbers = []
    for token in _NUMBER.findall(normalize(text)):
        try:
            numbers.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return numbers


def number_match(answer: str, expected, rel_tol: float = 0.005, abs_tol: float = 0.01) -> bool:
    target = float(expected)
    return any(math.isclose(n, target, rel_tol=rel_tol, abs_tol=abs_tol) for n in extract_numbers(answer))


def text_match(answer: str, expected) -> bool:
    return normalize(str(expected)).strip().lower() in normalize(answer).lower()


def list_match(answer: str, expected) -> bool:
    return all(text_match(answer, item) for item in expected)


def matches(answer: str, expected, kind: str) -> bool:
    if kind == "number":
        return number_match(answer, expected)
    if kind == "list":
        return list_match(answer, expected)
    return text_match(answer, expected)