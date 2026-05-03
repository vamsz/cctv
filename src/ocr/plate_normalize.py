"""Indian license plate format normalization and validation.

Standard format (Bharat-series and older state series both supported):

  Old format:  XX 00 XX 0000      e.g. TS 09 EA 1234   (state-RTO-series-number)
  New BH:      00 BH 0000 XX      e.g. 22 BH 1234 AA

OCR routinely confuses certain characters in the harsh roadside environment:

  O <-> 0     I <-> 1     S <-> 5     B <-> 8     Z <-> 2     G <-> 6

We apply position-aware corrections — at digit positions, force-convert
letters to their digit lookalike; at letter positions, the inverse.
"""
from __future__ import annotations

import re

VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN",
    "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS",
    "UK", "UP", "WB", "BH",  # BH = Bharat series
}

LETTER_TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                                  "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"})
DIGIT_TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})

OLD_FORMAT = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$")
BH_FORMAT  = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")


def _strip(plate: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", plate.upper())


def normalize_indian_plate(raw: str) -> str | None:
    """Best-effort normalization of an OCR string to a clean plate.

    Returns the canonical form (e.g. 'TS09EA1234') or None if even
    aggressive position-aware corrections can't shape it into a known
    Indian format.
    """
    s = _strip(raw)
    if not (8 <= len(s) <= 11):
        return None

    # Try old format first: 2 letters + 1-2 digits + 1-3 letters + 4 digits.
    # We don't know the exact split a priori, so try a few common splits.
    for split in _candidate_splits_old(s):
        try:
            state, rto, series, num = split
            state = state.translate(DIGIT_TO_LETTER)
            rto = rto.translate(LETTER_TO_DIGIT)
            series = series.translate(DIGIT_TO_LETTER)
            num = num.translate(LETTER_TO_DIGIT)
            candidate = f"{state}{rto}{series}{num}"
            if OLD_FORMAT.match(candidate) and state in VALID_STATE_CODES:
                return candidate
        except ValueError:
            continue

    # Try BH series: 2 digits + 'BH' + 4 digits + 1-2 letters.
    if len(s) >= 9 and "BH" in s:
        bh_idx = s.find("BH")
        if bh_idx == 2:
            year = s[0:2].translate(LETTER_TO_DIGIT)
            num = s[4:8].translate(LETTER_TO_DIGIT)
            tail = s[8:].translate(DIGIT_TO_LETTER)
            candidate = f"{year}BH{num}{tail}"
            if BH_FORMAT.match(candidate):
                return candidate

    return None


def _candidate_splits_old(s: str):
    n = len(s)
    # state always 2, num always 4 → middle (rto+series) is n-6.
    middle_len = n - 6
    if middle_len < 2 or middle_len > 5:
        return
    for rto_len in (1, 2):
        series_len = middle_len - rto_len
        if 1 <= series_len <= 3:
            yield s[0:2], s[2:2 + rto_len], s[2 + rto_len:2 + rto_len + series_len], s[-4:]


def is_valid_indian_plate(plate: str) -> bool:
    return normalize_indian_plate(plate) is not None
