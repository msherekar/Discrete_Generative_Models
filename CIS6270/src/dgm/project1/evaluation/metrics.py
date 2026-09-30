"""Sequence statistics shared by the CSV export and the figures.

Both read the same functions, so a number in a plot and the same number in a
CSV cannot drift apart.
"""
from collections import Counter
from itertools import combinations

import numpy as np


def _positional_entropy(sequences):
    if not sequences:
        return np.array([])
    L = len(sequences[0])
    entropy = []
    for pos in range(L):
        col = [s[pos] for s in sequences if pos < len(s)]
        counts = Counter(col)
        total  = len(col)
        probs  = np.array([v / total for v in counts.values()])
        h = -np.sum(probs * np.log2(probs + 1e-12))
        entropy.append(h)
    return np.array(entropy)


def _hamming(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))


def _pairwise_hamming(sequences):
    return [_hamming(a, b) for a, b in combinations(sequences, 2)] if len(sequences) >= 2 else [0]
