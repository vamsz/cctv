"""Indian license plate format normalization and validation.

Standard formats:
  Old format:  XX 00 XX 0000      e.g. TS 09 EA 1234   (state-RTO-series-number)
  New BH:      00 BH 0000 XX      e.g. 22 BH 1234 AA

OCR routinely confuses characters — this module applies position-aware
corrections to maximize valid plate extraction from noisy OCR output.

Aggressive approach: we try multiple interpretations of the raw string
and return the first one that matches a valid Indian plate pattern.
"""
from __future__ import annotations

import re

VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN",
    "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS",
    "UK", "UP", "WB", "AN",
    "BH",  # Bharat series
}

# OCR confusion pairs: character → what it should be at a digit position
LETTER_TO_DIGIT = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2", "R": "2",
    "E": "3",
    "A": "4", "H": "4",
    "S": "5",
    "G": "6", "C": "6",
    "T": "7", "Y": "7",
    "B": "8",
    "P": "9",
})

# OCR confusion pairs: digit → what it should be at a letter position
DIGIT_TO_LETTER = str.maketrans({
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
})

OLD_FORMAT = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$")
BH_FORMAT = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")


def _strip(plate: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", plate.upper())


def normalize_indian_plate(raw: str) -> str | None:
    """Best-effort normalization of an OCR string to a valid Indian plate.

    Tries multiple interpretations and corrections. Returns the canonical
    form (e.g. 'TS09EA1234') or None if nothing works.
    """
    if not raw:
        return None

    s = _strip(raw)
    if len(s) < 6 or len(s) > 13:
        return None

    # Try direct match first (no correction needed)
    direct = _try_parse(s)
    if direct:
        return direct

    # Try with position-aware corrections
    corrected = _try_corrected(s)
    if corrected:
        return corrected

    # Try removing common OCR artifacts (IND prefix, leading dots, etc.)
    for prefix in ("IND", "IN", "IMP"):
        if s.startswith(prefix) and len(s) > len(prefix) + 6:
            result = _try_corrected(s[len(prefix):])
            if result:
                return result

    # Try if there's a valid plate hiding inside a longer string
    if len(s) > 10:
        for start in range(len(s) - 8):
            sub = s[start:start + 10]
            result = _try_corrected(sub)
            if result:
                return result

    return None


def _try_parse(s: str) -> str | None:
    if OLD_FORMAT.match(s):
        state = s[:2]
        if state in VALID_STATE_CODES:
            return s
    if BH_FORMAT.match(s):
        return s
    return None


def _try_corrected(s: str) -> str | None:
    if len(s) < 6:
        return None

    # --- Old format attempts ---
    for split in _candidate_splits_old(s):
        state_raw, rto_raw, series_raw, num_raw = split
        state = state_raw.translate(DIGIT_TO_LETTER)
        rto = rto_raw.translate(LETTER_TO_DIGIT)
        series = series_raw.translate(DIGIT_TO_LETTER)
        num = num_raw.translate(LETTER_TO_DIGIT)
        candidate = f"{state}{rto}{series}{num}"
        if OLD_FORMAT.match(candidate) and state in VALID_STATE_CODES:
            return candidate

    # --- BH format attempts ---
    if len(s) >= 9:
        # Look for "BH" or near-BH in the string
        bh_positions = []
        for i in range(len(s) - 1):
            pair = s[i:i + 2]
            if pair == "BH":
                bh_positions.append(i)
            elif pair in ("8H", "BN", "B4"):
                bh_positions.append(i)

        for bh_idx in bh_positions:
            if bh_idx == 2 and len(s) >= 9:
                year = s[0:2].translate(LETTER_TO_DIGIT)
                num = s[4:8].translate(LETTER_TO_DIGIT) if len(s) >= 8 else ""
                tail = s[8:].translate(DIGIT_TO_LETTER) if len(s) > 8 else ""
                candidate = f"{year}BH{num}{tail}"
                if BH_FORMAT.match(candidate):
                    return candidate

    return None


def _candidate_splits_old(s: str):
    n = len(s)
    # state always 2 chars, number always 3-4 digits at end
    for num_len in (4, 3):
        middle_len = n - 2 - num_len
        if middle_len < 2 or middle_len > 5:
            continue
        for rto_len in (2, 1):
            series_len = middle_len - rto_len
            if 1 <= series_len <= 3:
                yield (
                    s[0:2],
                    s[2:2 + rto_len],
                    s[2 + rto_len:2 + rto_len + series_len],
                    s[n - num_len:],
                )


def is_valid_indian_plate(plate: str) -> bool:
    return normalize_indian_plate(plate) is not None


def format_display(plate: str) -> str:
    """Format a normalized plate for human display: 'TS09EA1234' → 'TS 09 EA 1234'."""
    m = OLD_FORMAT.match(plate)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"
    m = BH_FORMAT.match(plate)
    if m:
        return f"{m.group(1)} BH {m.group(2)} {m.group(3)}"
    return plate
