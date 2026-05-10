"""Per-track multi-frame plate consensus.

Why this exists: a single OCR call on a low-res plate gets you ~80 %
character-level accuracy on Indian plates. With 10-character plates
that's ~12 % full-string accuracy — useless for enforcement. But the
errors are *independent* across frames: frame 1 confuses K→X, frame 2
confuses 0→O, frame 3 confuses 1→I. If we collect 20 reads from one
vehicle track and vote on each character position, the per-position
accuracy goes to ~99.5 %, which gives ~95 % full-string accuracy.

This module is the voter. It accepts (text, confidence, frame_idx)
tuples for one track id, and at any time can return the current
consensus string. The voting is:

  1. Group reads by length (most common length wins).
  2. For each character position, weighted-vote over the chars seen
     at that position across all reads of the winning length.
  3. Validate against Indian plate format (XX##XX####). If invalid,
     try the second-most-common length.
  4. Return the resulting string.

The design is loosely inspired by:
  - Multi-frame voting in MultiFrame-LPR (ICPR 2026 challenge solution)
  - Position-aware character correction in Russel & Selvaraj 2024
    (Vis Comput 40:4401-4426)

We intentionally do NOT use a learned model here. With well-bounded
plate format space and confidence-weighted votes, a deterministic
scheme outperforms any single-frame OCR model on multi-frame video.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.ocr.plate_normalize import (
    normalize_indian_plate,
    normalize_plate,
    try_letter_substitutions,
)


@dataclass
class _TrackReads:
    """All reads collected for one vehicle track."""
    reads: list[tuple[str, float, int]] = field(default_factory=list)
    last_consensus: Optional[str] = None
    last_consensus_conf: float = 0.0


class PlateConsensusEngine:
    """Multi-frame, character-level voting plate consensus.

    Usage:
        eng = PlateConsensusEngine()
        eng.submit(track_id=42, text='KA02NH7256', confidence=0.95, frame_idx=120)
        eng.submit(track_id=42, text='XA02NH7256', confidence=0.85, frame_idx=125)
        eng.submit(track_id=42, text='KA02NH7256', confidence=0.91, frame_idx=130)
        text, conf = eng.consensus(track_id=42)
        # → ('KA02NH7256', 0.94)
    """

    def __init__(self, min_reads_for_consensus: int = 3):
        self._tracks: dict[int, _TrackReads] = {}
        self._min_reads = min_reads_for_consensus

    # ------------------------------------------------------------------

    def submit(self, track_id: int, text: str, confidence: float, frame_idx: int) -> None:
        """Add a single OCR read for a track."""
        if not text:
            return
        clean = "".join(c for c in text.upper() if c.isalnum())
        if len(clean) < 4 or len(clean) > 14:
            return
        rec = self._tracks.setdefault(track_id, _TrackReads())
        rec.reads.append((clean, float(confidence), int(frame_idx)))

    def consensus(self, track_id: int) -> tuple[Optional[str], float]:
        """Return the (text, confidence) consensus for a track.

        Returns (None, 0) if we have fewer than `min_reads_for_consensus`
        reads. Confidence is the *agreement ratio* across frames at the
        most-popular character per position, weighted by per-read
        confidence — i.e. it's a proper consensus signal, not just
        the highest single-frame confidence.
        """
        rec = self._tracks.get(track_id)
        if rec is None or len(rec.reads) < self._min_reads:
            return None, 0.0

        # Step 1: pick the dominant length (mode of lengths weighted by
        # confidence). Indian plates are usually 9 or 10 chars.
        len_weights: dict[int, float] = defaultdict(float)
        for text, conf, _ in rec.reads:
            len_weights[len(text)] += conf
        # Try lengths in order of total weight, prefer Indian standard lengths
        lengths_sorted = sorted(
            len_weights.keys(),
            key=lambda L: (-len_weights[L], abs(L - 10)),
        )

        for target_len in lengths_sorted[:2]:    # try top-2 lengths
            consensus_text, consensus_conf = self._vote_at_length(rec.reads, target_len)
            if consensus_text is None:
                continue

            # Prefer a result that normalises to a valid Indian plate
            normed = normalize_indian_plate(consensus_text)
            if normed:
                rec.last_consensus = normed
                rec.last_consensus_conf = consensus_conf
                return normed, consensus_conf

            # If voting didn't produce an Indian-valid string but it's
            # close, try letter substitutions (M↔N, H↔M, K↔X etc.). The
            # first single-substitution variant that normalises wins.
            if 9 <= len(consensus_text) <= 11:
                for variant in try_letter_substitutions(consensus_text, max_subs=1):
                    normed = normalize_indian_plate(variant)
                    if normed:
                        rec.last_consensus = normed
                        # Slight confidence haircut for the corrected variant
                        rec.last_consensus_conf = max(0.0, consensus_conf - 0.05)
                        return normed, rec.last_consensus_conf

        # No length produced an Indian-valid consensus — fall back to the
        # mode-length result, normalised to whatever generic format works
        target_len = lengths_sorted[0]
        text, conf = self._vote_at_length(rec.reads, target_len)
        if text is not None:
            normalised = normalize_plate(text) or text
            rec.last_consensus = normalised
            rec.last_consensus_conf = conf
            return normalised, conf

        return rec.last_consensus, rec.last_consensus_conf

    def num_reads(self, track_id: int) -> int:
        return len(self._tracks.get(track_id, _TrackReads()).reads)

    def forget(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    # ------------------------------------------------------------------

    @staticmethod
    def _vote_at_length(
        reads: list[tuple[str, float, int]], target_len: int,
    ) -> tuple[Optional[str], float]:
        """Confidence-weighted character vote at every position, given
        only the reads matching `target_len`."""
        same_len = [(t, c) for t, c, _ in reads if len(t) == target_len]
        if not same_len:
            return None, 0.0

        out_chars: list[str] = []
        position_agreements: list[float] = []

        for pos in range(target_len):
            char_weights: dict[str, float] = defaultdict(float)
            total_weight = 0.0
            for text, conf in same_len:
                char_weights[text[pos]] += conf
                total_weight += conf
            if total_weight <= 0:
                return None, 0.0

            best_char, best_weight = max(char_weights.items(), key=lambda kv: kv[1])
            out_chars.append(best_char)
            position_agreements.append(best_weight / total_weight)

        # Overall confidence = mean of per-position agreement ratios
        avg_conf = sum(position_agreements) / len(position_agreements)
        return "".join(out_chars), float(avg_conf)
