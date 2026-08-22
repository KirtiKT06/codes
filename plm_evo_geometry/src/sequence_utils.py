"""
sequence_utils.py

Sequence-level primitives: Hamming distance, BLOSUM-predicted distance (the
Q1 control), single/double mutant generation, and an efficient Hamming-1
neighbor graph builder that exploits the known structure of a DMS mutant
library instead of doing an O(n^2) all-pairs Hamming scan.

IMPORTANT NOTE ON NEIGHBOR GRAPHS (read before using robustness_evolvability.py):
For a *single-mutant-only* library, two single mutants at DIFFERENT positions
are Hamming distance 2 apart, not 1 -- so they are NOT neighbors of each other.
A mutant's Hamming-1 neighbors are only: (a) the wild type, and (b) other
single mutants AT THE SAME POSITION (different substituted residue). This
means single-mutant-only data gives each non-WT genotype a thin neighborhood
(<=19 possible neighbors, usually fewer since DMS libraries are rarely fully
saturated). Double-mutant data fills this out substantially: a double mutant
at positions (p, q) is a Hamming-1 neighbor of the two single mutants at p
and at q, and of other doubles sharing one position. build_neighbor_graph()
handles both cases correctly -- but robustness/evolvability estimates on
single-mutant-only data should be interpreted as lower-bound / sparse
estimates, not full neutral-network degree.
"""

from __future__ import annotations
import itertools
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

_BLOSUM = None
def _get_blosum(matrix_name: str = "BLOSUM62"):
    global _BLOSUM
    if _BLOSUM is None:
        from Bio.Align import substitution_matrices
        _BLOSUM = substitution_matrices.load(matrix_name)
    return _BLOSUM


def hamming_distance(seq1: str, seq2: str) -> int:
    if len(seq1) != len(seq2):
        raise ValueError(f"Sequences must be equal length ({len(seq1)} vs {len(seq2)})")
    return sum(a != b for a, b in zip(seq1, seq2))


def blosum_distance(seq1: str, seq2: str, matrix_name: str = "BLOSUM62") -> float:
    """
    Sum of -BLOSUM score over mismatched positions. Larger = more radical
    substitution (by substitution-matrix standards). Used as the Q1 control:
    regress embedding distance on this BEFORE comparing to raw Hamming
    distance, so you're testing what the PLM adds beyond substitution
    chemistry, not re-discovering BLOSUM.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be equal length")
    mat = _get_blosum(matrix_name)
    score = 0.0
    for a, b in zip(seq1, seq2):
        if a != b:
            try:
                s = mat[a, b]
            except KeyError:
                s = mat[b, a]
            score += -float(s)
    return score


@dataclass
class Variant:
    seq: str
    positions: tuple            # 0-indexed positions that differ from WT
    substitutions: tuple        # e.g. (('G', 5, 'A'),) for a single mutant
    dms_score: Optional[float] = None
    variant_id: Optional[int] = None


def generate_single_mutants(wt_seq: str, positions: Optional[list] = None,
                             exclude_wt_residue: bool = True) -> list[Variant]:
    """All 19 substitutions at every position (or a subset of positions)."""
    positions = positions if positions is not None else list(range(len(wt_seq)))
    variants = []
    for p in positions:
        wt_res = wt_seq[p]
        for aa in AMINO_ACIDS:
            if exclude_wt_residue and aa == wt_res:
                continue
            new_seq = wt_seq[:p] + aa + wt_seq[p + 1:]
            variants.append(Variant(seq=new_seq, positions=(p,),
                                     substitutions=((wt_res, p, aa),)))
    return variants


def generate_double_mutants(wt_seq: str, positions: Optional[list] = None,
                             max_pairs: Optional[int] = None) -> list[Variant]:
    """
    All pairwise double mutants across a set of positions. WARNING: this is
    combinatorial -- L positions * 19 * 19 substitutions / 2 grows fast
    (e.g. 56-position GB1 core -> ~56*55/2*361 ~ 550k variants). Use
    `positions` to restrict to a structurally meaningful subset (e.g. a
    binding interface) and/or `max_pairs` to randomly subsample position
    pairs before expanding substitutions.
    """
    positions = positions if positions is not None else list(range(len(wt_seq)))
    pos_pairs = list(itertools.combinations(positions, 2))
    if max_pairs is not None and len(pos_pairs) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pos_pairs), size=max_pairs, replace=False)
        pos_pairs = [pos_pairs[i] for i in idx]

    variants = []
    for p, q in pos_pairs:
        wt_p, wt_q = wt_seq[p], wt_seq[q]
        for aa_p in AMINO_ACIDS:
            if aa_p == wt_p:
                continue
            for aa_q in AMINO_ACIDS:
                if aa_q == wt_q:
                    continue
                new_seq = list(wt_seq)
                new_seq[p], new_seq[q] = aa_p, aa_q
                variants.append(Variant(seq="".join(new_seq), positions=(p, q),
                                         substitutions=((wt_p, p, aa_p), (wt_q, q, aa_q))))
    return variants


def build_neighbor_graph(wt_seq: str, variants: list[Variant]) -> dict[int, list[int]]:
    """
    Efficient Hamming-1 neighbor graph for a mutant library, exploiting
    known mutation positions rather than O(n^2) Hamming comparison.
    Index 0 = WT. Indices 1..N = variants, in input order.

    Rules (see module docstring):
      - WT is a neighbor of every single mutant.
      - Two singles are neighbors iff same position, different substitution.
      - A double mutant (p,q) is a neighbor of any single mutant that shares
        exactly one of its substitutions (at p or at q).
      - Two doubles are neighbors iff they share exactly one substitution.
    """
    n = len(variants)
    adj: dict[int, list[int]] = {i: [] for i in range(n + 1)}  # 0 = WT

    singles = [(i + 1, v) for i, v in enumerate(variants) if len(v.substitutions) == 1]
    doubles = [(i + 1, v) for i, v in enumerate(variants) if len(v.substitutions) == 2]

    # WT <-> singles
    for idx, v in singles:
        adj[0].append(idx)
        adj[idx].append(0)

    # singles <-> singles (same position)
    by_position: dict[int, list[tuple]] = {}
    for idx, v in singles:
        p = v.substitutions[0][1]
        by_position.setdefault(p, []).append((idx, v.substitutions[0]))
    for p, group in by_position.items():
        for (i1, s1), (i2, s2) in itertools.combinations(group, 2):
            adj[i1].append(i2)
            adj[i2].append(i1)

    # singles <-> doubles: a double containing exact substitution s as ONE of
    # its two mutations is ALWAYS a Hamming-1 neighbor of the single mutant
    # with substitution s, regardless of the double's other substitution
    # (the single is WT at the double's other position, so exactly one
    # position -- that other one -- differs). No extra filtering needed.
    sub_index: dict[tuple, list[int]] = {}  # substitution -> list of double idx containing it
    for idx, v in doubles:
        for s in v.substitutions:
            sub_index.setdefault(s, []).append(idx)

    for idx, v in singles:
        s = v.substitutions[0]
        for cand in sub_index.get(s, []):
            adj[idx].append(cand)
            adj[cand].append(idx)

    # doubles <-> doubles: TWO DOUBLES ARE HAMMING-1 NEIGHBORS ONLY IF THEY
    # MUTATE THE EXACT SAME PAIR OF POSITIONS, differing in the residue at
    # exactly ONE of the two positions. Sharing a single substitution while
    # the *other* mutated position differs is NOT sufficient -- that case is
    # Hamming distance 2 or 3, not 1 (both "other" positions differ from
    # each other's WT baseline). Group by position-pair first, then compare
    # only within each group.
    by_pos_pair: dict[tuple, list[tuple]] = {}
    for idx, v in doubles:
        key = tuple(sorted(v.positions))
        by_pos_pair.setdefault(key, []).append((idx, v.substitutions))

    for key, group in by_pos_pair.items():
        for (i1, subs1), (i2, subs2) in itertools.combinations(group, 2):
            # subs are ordered by position already (positions sorted at generation);
            # align by position to compare residues at each shared position.
            d1 = {pos: aa for (_, pos, aa) in subs1}
            d2 = {pos: aa for (_, pos, aa) in subs2}
            n_diff = sum(1 for p in key if d1[p] != d2[p])
            if n_diff == 1:
                adj[i1].append(i2)
                adj[i2].append(i1)

    for k in adj:
        adj[k] = sorted(set(adj[k]))
    return adj
