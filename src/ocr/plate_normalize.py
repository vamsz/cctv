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
PARTIAL_OLD_FORMAT = re.compile(r"^([A-Z]{1,2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$")
BH_FORMAT = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")


def _strip(plate: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", plate.upper())


def normalize_indian_plate(raw: str) -> str | None:
    """Best-effort normalization of an OCR string to a valid Indian plate.

    Tries multiple interpretations and corrections in priority order:
      1. Direct match (no correction)
      2. Position-aware char confusion correction
      3. Artifact stripping (IND prefix etc.)
      4. Substring search in longer strings
      5. Beam-search over confusion candidates (edit distance ≤ 2)

    Returns the canonical form (e.g. 'TS09EA1234') or None.
    """
    if not raw:
        return None

    s = _strip(raw)
    if len(s) < 6 or len(s) > 13:
        return None

    # 1. Direct match
    direct = _try_parse(s)
    if direct:
        return direct

    # 2. Position-aware corrections
    corrected = _try_corrected(s)
    if corrected:
        return corrected

    # 3. Artifact stripping
    for prefix in ("IND", "IN", "IMP"):
        if s.startswith(prefix) and len(s) > len(prefix) + 6:
            result = _try_corrected(s[len(prefix):])
            if result:
                return result

    # 4. Substring search
    if len(s) > 10:
        for start in range(len(s) - 8):
            sub = s[start:start + 10]
            result = _try_corrected(sub)
            if result:
                return result

    # 5. Beam search over confusion candidates (catches multi-char OCR errors)
    bs = _beam_search_normalize(s, beam_width=8)
    if bs:
        return bs

    # 6. State-code edit-distance correction (UNAMBIGUOUS only).
    # If the first 2 characters look like a state code with a small OCR
    # error and there is EXACTLY ONE valid state code at edit distance
    # 1, correct it. If multiple states are equally close (e.g. XA is
    # 1 hop from both KA and GA), do NOT auto-correct — leave the
    # original text so the multi-frame consensus engine can decide
    # via majority voting across many reads of the same vehicle.
    if len(s) >= 8:
        prefix = s[:2]
        candidates_at_d1 = [
            state for state in VALID_STATE_CODES
            if state != prefix and _hamming_within_1(prefix, state)
        ]
        # ONE unambiguous candidate → apply it.
        if len(candidates_at_d1) == 1:
            fixed = candidates_at_d1[0] + s[2:]
            result = _try_parse(fixed) or _try_corrected(fixed)
            if result:
                return result
        # Multiple candidates → cross-reference with visually-likely OCR
        # confusion pairs at position 0. K↔X, M↔N, S↔5, B↔8 etc. Only
        # accept the candidate that's reachable through a documented
        # confusion at exactly position 0, ignoring the rest.
        elif len(candidates_at_d1) > 1:
            for state in candidates_at_d1:
                if _is_known_ocr_confusion(prefix[0], state[0]) and prefix[1] == state[1]:
                    fixed = state + s[2:]
                    result = _try_parse(fixed) or _try_corrected(fixed)
                    if result:
                        return result
    return None


# Symmetric letter-letter OCR confusions that fire at ANY position
# inside the alphabetic portions of an Indian plate (positions 0-1 for
# the state code and 4-5 for the series). Curated from PARSeq/ABINet
# error tables. Used by `try_letter_substitutions` below.
_OCR_LETTER_CONFUSIONS: dict[str, list[str]] = {
    "M": ["N", "H", "W"],
    "N": ["M", "H"],
    "H": ["N", "M", "K"],
    "K": ["X", "H", "R"],
    "X": ["K"],
    "B": ["R", "P", "8"],
    "R": ["B", "P", "K"],
    "P": ["R", "B", "F"],
    "F": ["P", "E"],
    "E": ["F", "B"],
    "G": ["C", "Q", "6"],
    "C": ["G", "Q"],
    "Q": ["G", "C", "O"],
    "O": ["Q", "D", "0"],
    "D": ["O", "0"],
    "U": ["V"],
    "V": ["U", "Y"],
    "Y": ["V", "T"],
    "T": ["I", "Y"],
    "I": ["T", "L", "1", "J"],
    "L": ["I", "1"],
    "J": ["I", "1"],
    "S": ["5", "8"],
    "Z": ["2"],
    "0": ["O", "Q", "D"],
    "1": ["I", "L", "J"],
    "2": ["Z"],
    "3": ["E"],
    "4": ["A", "H"],
    "5": ["S"],
    "6": ["G", "C"],
    "7": ["T", "Y"],
    "8": ["B", "S"],
    "9": ["P"],
}


def try_letter_substitutions(text: str, max_subs: int = 1) -> list[str]:
    """Generate variants of `text` by substituting confusable letters.

    Used to recover from M↔N, H↔M and similar non-state-code mistakes.
    Returns variants up to `max_subs` substitutions away. Only letter
    positions are touched; digit positions are preserved.
    """
    variants = [text]
    if max_subs <= 0:
        return variants
    out: list[str] = []
    for v in variants:
        for i, ch in enumerate(v):
            if not ch.isalnum():
                continue
            for repl in _OCR_LETTER_CONFUSIONS.get(ch, ()):
                out.append(v[:i] + repl + v[i+1:])
    if max_subs > 1:
        # 2-substitution variants
        for v in list(out):
            for i, ch in enumerate(v):
                if not ch.isalnum():
                    continue
                for repl in _OCR_LETTER_CONFUSIONS.get(ch, ()):
                    out.append(v[:i] + repl + v[i+1:])
    # Dedupe, drop the input itself (caller already tried it)
    seen = {text}
    deduped = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        deduped.append(v)
    return deduped


# Known OCR confusion pairs at position 0 of an Indian plate. Curated
# from common scene-text-recognition error tables (PARSeq, ABINet) plus
# typical CCTV / fast-alpr failure modes on Indian fonts.
_OCR_CONFUSION_AT_0: set[tuple[str, str]] = {
    # X ↔ K — fast-alpr reads K's diagonal stroke as an X
    ("X", "K"), ("K", "X"),
    # M ↔ N — center-bar misses
    ("M", "N"), ("N", "M"),
    # B ↔ 8 / B ↔ R
    ("B", "R"), ("R", "B"), ("B", "8"), ("8", "B"),
    # T ↔ I when middle stroke faded
    ("T", "I"), ("I", "T"),
    # 0 ↔ O ↔ Q ↔ D
    ("0", "O"), ("O", "0"), ("D", "0"), ("D", "O"), ("O", "D"),
    # 5 ↔ S
    ("S", "5"), ("5", "S"),
    # 1 ↔ I ↔ L ↔ J
    ("I", "1"), ("1", "I"), ("L", "I"), ("I", "L"), ("J", "I"),
    # G ↔ C ↔ 6
    ("G", "C"), ("C", "G"), ("G", "6"), ("6", "G"),
    # 2 ↔ Z
    ("Z", "2"), ("2", "Z"),
    # H ↔ N (vertical bars only)
    ("H", "N"), ("N", "H"),
    # P ↔ R / P ↔ B
    ("P", "R"), ("R", "P"), ("P", "B"), ("B", "P"),
    # U ↔ V
    ("U", "V"), ("V", "U"),
}


def _is_known_ocr_confusion(observed: str, candidate: str) -> bool:
    return (observed, candidate) in _OCR_CONFUSION_AT_0


def _hamming_within_1(a: str, b: str) -> bool:
    """True iff strings of equal length differ by ≤ 1 character."""
    if len(a) != len(b):
        return False
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    return diffs <= 1


def _beam_search_normalize(s: str, beam_width: int = 8) -> str | None:
    """Try all single + double character confusion substitutions.

    Generates candidate strings by replacing characters with their
    confusion pairs (O↔0, I↔1, S↔5, B↔8, Z↔2, E↔3, A↔4, G↔6, T↔7).
    Uses a beam to limit the search to the top `beam_width` candidates.
    """
    # All confusion pairs (bidirectional)
    CONFUSIONS: dict[str, list[str]] = {
        "O": ["0"], "0": ["O"],
        "I": ["1", "L"], "1": ["I", "L"], "L": ["1", "I"],
        "S": ["5"], "5": ["S"],
        "B": ["8"], "8": ["B"],
        "Z": ["2"], "2": ["Z"],
        "E": ["3"], "3": ["E"],
        "A": ["4"], "4": ["A"],
        "G": ["6"], "6": ["G"],
        "T": ["7"], "7": ["T"],
        "Q": ["0"], "D": ["0"],
        "J": ["1"], "R": ["2"],
        "C": ["6"], "Y": ["7"],
        "H": ["4"], "P": ["9"], "9": ["P"],
    }

    # Start with the original string as the only candidate
    # Score = number of substitutions (lower is better)
    beam: list[tuple[int, str]] = [(0, s)]
    seen: set[str] = {s}
    best: str | None = None

    for _depth in range(min(len(s), 4)):   # max 4 substitutions
        next_beam: list[tuple[int, str]] = []
        for cost, candidate in beam:
            for i, ch in enumerate(candidate):
                for replacement in CONFUSIONS.get(ch, []):
                    new_s = candidate[:i] + replacement + candidate[i + 1:]
                    if new_s in seen:
                        continue
                    seen.add(new_s)
                    result = _try_corrected(new_s)
                    if result:
                        if best is None:
                            best = result  # first valid hit wins
                        continue
                    next_beam.append((cost + 1, new_s))
        if best:
            return best
        # Keep best beam_width candidates (fewest substitutions)
        beam = sorted(next_beam, key=lambda x: x[0])[:beam_width]
        if not beam:
            break

    return best


def _try_parse(s: str) -> str | None:
    if OLD_FORMAT.match(s):
        state = s[:2]
        if state in VALID_STATE_CODES:
            return s
    if BH_FORMAT.match(s):
        return s
    return None


def _try_corrected(s: str) -> str | None:
    if len(s) < 4:
        return None

    # --- Old format attempts ---
    best_candidate = None
    for split in _candidate_splits_old(s):
        state_raw, rto_raw, series_raw, num_raw = split
        state = state_raw.translate(DIGIT_TO_LETTER)
        rto = rto_raw.translate(LETTER_TO_DIGIT)
        series = series_raw.translate(DIGIT_TO_LETTER)
        num = num_raw.translate(LETTER_TO_DIGIT)
        candidate = f"{state}{rto}{series}{num}"
        
        if OLD_FORMAT.match(candidate) and state in VALID_STATE_CODES:
            return candidate
            
        if PARTIAL_OLD_FORMAT.match(candidate):
            if best_candidate is None:
                best_candidate = candidate

    if best_candidate:
        return best_candidate

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
    for num_len in range(1, 5):
        for series_len in range(1, 4):
            for rto_len in range(1, 3):
                state_len = n - (num_len + series_len + rto_len)
                if 1 <= state_len <= 2:
                    yield (
                        s[0:state_len],
                        s[state_len:state_len + rto_len],
                        s[state_len + rto_len:state_len + rto_len + series_len],
                        s[n - num_len:],
                    )


# ---------------------------------------------------------------------------
# International plate normalization
# ---------------------------------------------------------------------------

# UK format: AB12CDE  (2 alpha + 2 digit + 3 alpha, 7 chars)
_UK_FORMAT = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$")

# EU/generic: 4-12 alphanumeric chars (good-faith catch-all for non-Indian plates)
_GENERIC_MIN = 4
_GENERIC_MAX = 12


def normalize_plate(raw: str) -> str | None:
    """Normalize any plate: tries Indian first, then UK, then generic alphanumeric.

    Returns cleaned text if the raw string plausibly represents a plate,
    or None when it is clearly garbage (< 4 or > 12 alphanumeric chars).
    """
    if not raw:
        return None

    # Try Indian format (strictest — position-aware corrections + validation)
    indian = normalize_indian_plate(raw)
    if indian:
        return indian

    # Clean to alphanumeric only
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())

    if len(s) < _GENERIC_MIN or len(s) > _GENERIC_MAX:
        return None

    # UK format exact match
    if _UK_FORMAT.match(s):
        return s

    # Generic: accept any 4-12 alphanumeric string (handles EU, US, etc.)
    return s


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
