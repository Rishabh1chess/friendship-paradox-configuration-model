"""Run every experiment, write results/results.json and the two figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neighbor_degree as nd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED = 168
N_MAIN = 100_000
SCALE_N = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000]
REPS = 16

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"

LAWS = {
    "poisson": nd.poisson_pmf(4.0, 40),
    "powerlaw": nd.power_law_pmf(2.5, 2, 300),
}
LABELS = {"poisson": "Poisson, mean 4", "powerlaw": r"Power law, $q_k \propto k^{-2.5}$, $2 \leq k \leq 300$"}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "axes.grid": False,
            "font.size": 10.5,
            "legend.frameon": False,
        }
    )


def share_below_neighbours(edges: np.ndarray, deg: np.ndarray) -> float:
    """Share of nodes (with degree > 0) whose mean neighbour degree exceeds their own."""
    n = len(deg)
    total = np.bincount(edges[:, 0], weights=deg[edges[:, 1]], minlength=n)
    total += np.bincount(edges[:, 1], weights=deg[edges[:, 0]], minlength=n)
    has = deg > 0
    return float(np.mean(total[has] / deg[has] > deg[has]))


def main_experiment(rng: np.random.Generator) -> tuple[dict, dict]:
    res: dict = {"n_main": N_MAIN, "seed": SEED}
    empirical: dict = {}
    for name, q in LAWS.items():
        deg = nd.sample_degrees(q, N_MAIN, rng)
        edges = nd.configuration_model(deg, rng)
        emp = nd.empirical_pmf(nd.edge_end_degrees(edges, deg), len(q))
        empirical[name] = emp
        res[f"{name}_mean_degree_theory"] = round(nd.mean_degree(q), 4)
        res[f"{name}_mean_degree_observed"] = round(float(deg.mean()), 4)
        res[f"{name}_neighbour_mean_theory"] = round(nd.neighbour_mean(q), 4)
        res[f"{name}_neighbour_mean_observed"] = round(float(nd.edge_end_degrees(edges, deg).mean()), 4)
        res[f"{name}_tv_vs_plain_degree_law"] = round(nd.total_variation(emp, q), 4)
        res[f"{name}_tv_vs_size_biased_law"] = round(nd.total_variation(emp, nd.size_biased(q)), 4)
        res[f"{name}_share_nodes_below_their_neighbours"] = round(share_below_neighbours(edges, deg), 4)
    return res, empirical


def scaling_experiment(rng: np.random.Generator) -> dict:
    out: dict = {"n": SCALE_N, "reps": REPS}
    for name, q in LAWS.items():
        tri, frac, tv_sb, tv_plain = [], [], [], []
        for n in SCALE_N:
            t, f, a, b = [], [], [], []
            for _ in range(REPS):
                deg = nd.sample_degrees(q, n, rng)
                edges = nd.configuration_model(deg, rng)
                count, on_tri = nd.triangle_stats(nd.simple_adjacency(edges, n))
                emp = nd.empirical_pmf(nd.edge_end_degrees(edges, deg), len(q))
                t.append(count)
                f.append(on_tri / n)
                a.append(nd.total_variation(emp, nd.size_biased(q)))
                b.append(nd.total_variation(emp, q))
            tri.append(float(np.mean(t)))
            frac.append(float(np.mean(f)))
            tv_sb.append(float(np.mean(a)))
            tv_plain.append(float(np.mean(b)))
        out[f"{name}_triangles_mean"] = [round(x, 3) for x in tri]
        out[f"{name}_share_nodes_on_triangle"] = [round(x, 6) for x in frac]
        out[f"{name}_tv_size_biased"] = [round(x, 4) for x in tv_sb]
        out[f"{name}_tv_plain"] = [round(x, 4) for x in tv_plain]
        out[f"{name}_triangles_theory"] = round(nd.expected_triangles(q), 3)
    return out


def hero_figure(empirical: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, name in zip(axes, LAWS):
        q = LAWS[name]
        emp = empirical[name]
        ks = np.arange(len(q))
        hi = 16 if name == "poisson" else 60
        sl = slice(0 if name == "poisson" else 2, hi + 1)
        ax.plot(ks[sl], q[sl], "o--", color=GREY, ms=4.5, lw=1, label=r"plain degree law $q_k$")
        ax.plot(ks[sl], nd.size_biased(q)[sl], "-", color=BLUE, lw=2, label=r"prediction $k\,q_k/\langle k\rangle$")
        ax.plot(ks[sl], emp[: len(q)][sl], "o", color=ORANGE, ms=4.5, label="measured, ends of edges")
        ax.set_xlabel("degree k of the node at the end of an edge")
        ax.set_ylabel("probability")
        ax.set_title(LABELS[name], fontsize=10.5)
        if name == "powerlaw":
            ax.set_xscale("log")
            ax.set_yscale("log")
            ticks = [2, 3, 5, 10, 20, 50]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.minorticks_off()
        ax.legend(loc="upper right" if name == "poisson" else "lower left")
    fig.suptitle(f"A random neighbour is not a random node (configuration model, n = {N_MAIN:,})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "neighbour_degree.png", dpi=160)
    plt.close(fig)


def scaling_figure(sc: dict) -> None:
    n = np.array(sc["n"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    cols = {"poisson": BLUE, "powerlaw": ORANGE}
    names = {"poisson": "Poisson, mean 4", "powerlaw": "power law 2.5"}

    ax = axes[0]
    for k, c in cols.items():
        ax.plot(n, sc[f"{k}_triangles_mean"], "o-", color=c, label=names[k])
        ax.axhline(sc[f"{k}_triangles_theory"], color=c, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("number of nodes n")
    ax.set_ylabel("triangles in the graph (mean of runs)")
    ax.set_title("Triangle count levels off (dotted = formula)", fontsize=10.5)
    ax.legend()

    ax = axes[1]
    for k, c in cols.items():
        ax.plot(n, sc[f"{k}_share_nodes_on_triangle"], "o-", color=c, label=names[k])
    ax.plot(n, sc["poisson_share_nodes_on_triangle"][0] * n[0] / n, color=GREY, ls="--", lw=1, label=r"$\propto 1/n$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of nodes n")
    ax.set_ylabel("share of nodes on a triangle")
    ax.set_title("So almost no node sees a cycle", fontsize=10.5)
    ax.legend()

    ax = axes[2]
    for k, c in cols.items():
        ax.plot(n, sc[f"{k}_tv_size_biased"], "o-", color=c, label=f"{names[k]}: vs $k q_k/\\langle k\\rangle$")
        ax.plot(n, sc[f"{k}_tv_plain"], "s--", color=c, alpha=0.55, label=f"{names[k]}: vs plain $q_k$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of nodes n")
    ax.set_ylabel("total variation distance")
    ax.set_title("Neighbour law converges to the size-biased one", fontsize=10.5)
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=160)
    plt.close(fig)


def main() -> None:
    if not OUT.exists():
        OUT.mkdir(parents=True)
    style()
    rng = np.random.default_rng(SEED)
    results, empirical = main_experiment(rng)
    results["scaling"] = scaling_experiment(rng)
    results["largest_n"] = SCALE_N[-1]
    for name in LAWS:
        results[f"{name}_triangles_at_largest_n"] = results["scaling"][f"{name}_triangles_mean"][-1]
        results[f"{name}_triangles_theory"] = results["scaling"][f"{name}_triangles_theory"]
        results[f"{name}_share_on_triangle_at_largest_n"] = results["scaling"][f"{name}_share_nodes_on_triangle"][-1]
    hero_figure(empirical)
    scaling_figure(results["scaling"])
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if k != "scaling":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
