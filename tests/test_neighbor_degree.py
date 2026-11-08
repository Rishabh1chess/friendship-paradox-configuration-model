import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import neighbor_degree as nd  # noqa: E402


class TestDistributions(unittest.TestCase):
    def test_pmfs_sum_to_one(self):
        self.assertAlmostEqual(nd.poisson_pmf(4.0, 60).sum(), 1.0)
        self.assertAlmostEqual(nd.power_law_pmf(2.5, 2, 100).sum(), 1.0)
        q = nd.poisson_pmf(3.0, 50)
        self.assertAlmostEqual(nd.size_biased(q).sum(), 1.0)
        self.assertAlmostEqual(nd.excess(q).sum(), 1.0, places=9)

    def test_size_biased_mean_is_ratio_of_moments(self):
        q = nd.power_law_pmf(2.5, 2, 200)
        sb = nd.size_biased(q)
        mean_sb = float(np.dot(np.arange(len(sb)), sb))
        self.assertAlmostEqual(mean_sb, nd.second_moment(q) / nd.mean_degree(q))
        self.assertAlmostEqual(mean_sb, nd.neighbour_mean(q))

    def test_poisson_excess_degree_is_poisson(self):
        # Known identity: the excess law of a Poisson degree law is the same Poisson.
        lam = 3.5
        q = nd.poisson_pmf(lam, 200)
        ex = nd.excess(q)
        ref = nd.poisson_pmf(lam, 200)[: len(ex)]
        self.assertLess(np.abs(ex - ref).max(), 1e-12)

    def test_friendship_paradox_never_negative(self):
        # <k^2>/<k> >= <k> always, with equality only for a constant degree.
        for q in (nd.poisson_pmf(2.0, 40), nd.power_law_pmf(3.0, 1, 80)):
            self.assertGreater(nd.neighbour_mean(q), nd.mean_degree(q))
        regular = np.zeros(6)
        regular[3] = 1.0
        self.assertAlmostEqual(nd.neighbour_mean(regular), 3.0)

    def test_total_variation_basics(self):
        p = np.array([0.5, 0.5])
        self.assertEqual(nd.total_variation(p, p), 0.0)
        self.assertAlmostEqual(nd.total_variation(np.array([1.0]), np.array([0.0, 1.0])), 1.0)


class TestConfigurationModel(unittest.TestCase):
    def test_stub_pairing_preserves_degrees(self):
        rng = np.random.default_rng(1)
        deg = nd.sample_degrees(nd.poisson_pmf(3.0, 30), 2000, rng)
        edges = nd.configuration_model(deg, rng)
        self.assertEqual(len(edges), deg.sum() // 2)
        realised = np.bincount(edges.ravel(), minlength=len(deg))
        self.assertTrue(np.array_equal(realised, deg))

    def test_odd_degree_sum_rejected(self):
        with self.assertRaises(ValueError):
            nd.configuration_model(np.array([1, 1, 1]), np.random.default_rng(0))

    def test_neighbour_degree_matches_networkx(self):
        # Average degree of edge ends = sum_i k_i^2 / sum_i k_i, which is also
        # the degree-weighted mean of networkx's average_neighbor_degree.
        g = nx.gnp_random_graph(500, 0.02, seed=3)
        edges = np.array(list(g.edges()))
        deg = np.array([g.degree(i) for i in range(500)])
        mine = nd.edge_end_degrees(edges, deg).mean()
        and_ = nx.average_neighbor_degree(g)
        theirs = sum(deg[i] * and_[i] for i in g) / deg.sum()
        self.assertAlmostEqual(mine, theirs)
        self.assertAlmostEqual(mine, (deg**2).sum() / deg.sum())

    def test_sampled_neighbour_law_close_to_prediction(self):
        rng = np.random.default_rng(4)
        q = nd.poisson_pmf(4.0, 40)
        deg = nd.sample_degrees(q, 40000, rng)
        edges = nd.configuration_model(deg, rng)
        emp = nd.empirical_pmf(nd.edge_end_degrees(edges, deg), len(q))
        self.assertLess(nd.total_variation(emp, nd.size_biased(q)), 0.02)
        self.assertGreater(nd.total_variation(emp, q), 0.1)


class TestTreeLikeness(unittest.TestCase):
    def test_single_triangle(self):
        adj = nd.simple_adjacency(np.array([[0, 1], [1, 2], [2, 0]]), 3)
        self.assertEqual(nd.triangle_stats(adj), (1, 3))

    def test_tree_has_no_triangles(self):
        t = nx.balanced_tree(2, 4)
        adj = sparse.csr_matrix(nx.to_scipy_sparse_array(t))
        self.assertEqual(nd.triangle_stats(adj), (0, 0))

    def test_triangles_match_networkx(self):
        g = nx.gnp_random_graph(300, 0.04, seed=5)
        edges = np.array(list(g.edges()))
        adj = nd.simple_adjacency(edges, 300)
        count, on_tri = nd.triangle_stats(adj)
        tri = nx.triangles(g)
        self.assertEqual(count, sum(tri.values()) // 3)
        self.assertEqual(on_tri, sum(1 for v in tri.values() if v > 0))

    def test_simple_adjacency_drops_loops_and_repeats(self):
        adj = nd.simple_adjacency(np.array([[0, 0], [0, 1], [1, 0], [0, 1]]), 2)
        self.assertEqual(adj.nnz, 2)
        self.assertEqual(adj.max(), 1.0)

    def test_expected_triangles_poisson_closed_form(self):
        # For Poisson(lam), (<k^2>-<k>)^3 / (6 <k>^3) = lam^3 / 6.
        q = nd.poisson_pmf(4.0, 120)
        self.assertAlmostEqual(nd.expected_triangles(q), 4.0**3 / 6.0, places=6)


if __name__ == "__main__":
    unittest.main()
