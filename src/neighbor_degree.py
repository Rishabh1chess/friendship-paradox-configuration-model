"""Neighbour degrees and local tree-likeness in the configuration model.

Everything here is built from scratch on numpy/scipy; networkx is used only in
the tests as an independent check.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import poisson


# ---------------------------------------------------------------- distributions
def poisson_pmf(mean: float, kmax: int) -> np.ndarray:
    """Poisson pmf on 0..kmax, renormalised so the truncated vector sums to 1."""
    q = poisson.pmf(np.arange(kmax + 1), mean)
    return q / q.sum()


def power_law_pmf(exponent: float, kmin: int, kmax: int) -> np.ndarray:
    """q_k proportional to k^-exponent on kmin..kmax, zero elsewhere (index = k)."""
    k = np.arange(kmax + 1, dtype=float)
    q = np.zeros(kmax + 1)
    q[kmin:] = k[kmin:] ** -exponent
    return q / q.sum()


def mean_degree(q: np.ndarray) -> float:
    """<k> = sum_k k q_k."""
    return float(np.dot(np.arange(len(q)), q))


def second_moment(q: np.ndarray) -> float:
    """<k^2> = sum_k k^2 q_k."""
    k = np.arange(len(q))
    return float(np.dot(k * k, q))


def size_biased(q: np.ndarray) -> np.ndarray:
    """Degree law of the node at the end of a uniformly random edge: k q_k / <k>."""
    k = np.arange(len(q))
    return k * q / mean_degree(q)


def excess(q: np.ndarray) -> np.ndarray:
    """Law of the *other* edges at such a node: (k+1) q_{k+1} / <k>, on 0..kmax-1."""
    return size_biased(q)[1:]


def neighbour_mean(q: np.ndarray) -> float:
    """Mean degree of a random neighbour, which works out to <k^2> / <k>."""
    return second_moment(q) / mean_degree(q)


def total_variation(p: np.ndarray, r: np.ndarray) -> float:
    """Total variation distance between two pmfs, padding the shorter with zeros."""
    m = max(len(p), len(r))
    a = np.zeros(m)
    b = np.zeros(m)
    a[: len(p)] = p
    b[: len(r)] = r
    return float(0.5 * np.abs(a - b).sum())


# ------------------------------------------------------------ the random graph
def sample_degrees(q: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n i.i.d. degrees from q, redrawing single nodes until the sum is even."""
    ks = np.arange(len(q))
    deg = rng.choice(ks, size=n, p=q)
    while deg.sum() % 2:
        deg[rng.integers(n)] = rng.choice(ks, p=q)
    return deg


def configuration_model(deg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pair stubs uniformly at random. Returns an (m, 2) array of edges.

    Self-loops and multi-edges are kept, as in the textbook model.
    """
    if deg.sum() % 2:
        raise ValueError("degree sum must be even")
    stubs = rng.permutation(np.repeat(np.arange(len(deg)), deg))
    return stubs.reshape(-1, 2)


def edge_end_degrees(edges: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """Degree of the node reached by following each edge in each direction."""
    return deg[edges.ravel()]


def empirical_pmf(values: np.ndarray, length: int | None = None) -> np.ndarray:
    """Histogram of non-negative integers as a pmf."""
    counts = np.bincount(values, minlength=length or 0)
    return counts / counts.sum()


def simple_adjacency(edges: np.ndarray, n: int) -> sparse.csr_matrix:
    """Symmetric 0/1 adjacency with self-loops and repeated edges removed."""
    keep = edges[:, 0] != edges[:, 1]
    e = edges[keep]
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    a = (a + a.T).tocsr()
    a.data[:] = 1.0
    return a


# ------------------------------------------------------------ tree-likeness
def triangle_stats(adj: sparse.csr_matrix) -> tuple[int, int]:
    """(number of triangles, number of nodes lying on at least one triangle)."""
    closed = (adj @ adj).multiply(adj)
    per_node = np.asarray(closed.sum(axis=1)).ravel() / 2.0
    return int(round(per_node.sum() / 3.0)), int((per_node > 0).sum())


def expected_triangles(q: np.ndarray) -> float:
    """Large-n triangle count of the configuration model:
    (<k^2> - <k>)^3 / (6 <k>^3).
    """
    k1 = mean_degree(q)
    return (second_moment(q) - k1) ** 3 / (6.0 * k1**3)
