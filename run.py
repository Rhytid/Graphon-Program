"""Sample graphs from a graphon and study how their eigenvalues behave.

Example:
    python run.py --formula "x*y" --n 100 200 400 800 --trials 20

Outputs (in --out, default results/):
    eigenvalues.csv         every eigenvalue of every sampled graph (n, trial, index, eigenvalue, eigenvalue/n)
    multiplicities_n<N>.txt eigenvalue list with multiplicity for the first sample at each n
    summary.csv             mean/std of the top-k normalized eigenvalues per n, next to the graphon limit
    graphon.png             heatmap of W(x,y)
    top_eigenvalues.png     top-k eigenvalue/n vs n, converging to the graphon operator eigenvalues
    spectrum_n<N>.png       histogram of all eigenvalues of the largest graphs
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graphon_lab import Graphon, eigenvalues, sample_adjacency, with_multiplicity
from graphon_lab.sampling import METHODS
from graphon_lab.spectrum import format_multiplicities


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--formula", default="x*y", help='graphon W(x,y) in x and y, e.g. "x*y", "exp(-abs(x-y))"')
    p.add_argument("--n", type=int, nargs="+", default=[100, 200, 400, 800], help="graph sizes to sample")
    p.add_argument("--trials", type=int, default=20, help="number of sampled graphs per size")
    p.add_argument("--method", choices=METHODS, default="random", help="how latent points are chosen")
    p.add_argument("--top-k", type=int, default=5, help="how many leading eigenvalues to track")
    p.add_argument("--tol", type=float, default=1e-8, help="tolerance for grouping equal eigenvalues")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    W = Graphon(args.formula)
    limit = W.operator_eigenvalues()[: args.top_k]
    print(f"Graphon W(x,y) = {W.formula}   sampling = {args.method}   seed = {args.seed}")
    print(f"Graphon operator top-{args.top_k} eigenvalues (limit of eig/n): {np.round(limit, 5)}")

    top = {}  # n -> array (trials, top_k) of eigenvalue / n
    last_spectra = None
    with open(out / "eigenvalues.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "trial", "index", "eigenvalue", "eigenvalue_over_n"])
        for n in args.n:
            rows = []
            spectra = []
            for t in range(args.trials):
                A, _ = sample_adjacency(W, n, args.method, rng)
                eigs = eigenvalues(A)
                spectra.append(eigs)
                writer.writerows([n, t, i, e, e / n] for i, e in enumerate(eigs))
                rows.append(eigs[: args.top_k] / n)
                if t == 0:
                    groups = with_multiplicity(eigs, args.tol)
                    (out / f"multiplicities_n{n}.txt").write_text(
                        f"W(x,y) = {W.formula}, n = {n}, trial 0, tol = {args.tol}\n"
                        f"{len(groups)} distinct eigenvalues out of {n}\n\n"
                        + format_multiplicities(groups) + "\n"
                    )
            top[n] = np.array(rows)
            last_spectra = (n, np.concatenate(spectra))

    write_summary(out, args, top, limit)
    plot_graphon(out, W)
    plot_top(out, args, top, limit)
    plot_spectrum(out, W, *last_spectra)
    print(f"\nWrote results to {out.resolve()}")


def write_summary(out, args, top, limit):
    print(f"\n{'n':>6} " + " ".join(f"{'λ' + str(i + 1) + '/n':>18}" for i in range(args.top_k)))
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "index", "mean_eig_over_n", "std_eig_over_n", "graphon_limit", "abs_error"])
        for n, vals in top.items():
            mean, std = vals.mean(axis=0), vals.std(axis=0)
            cells = []
            for i in range(args.top_k):
                writer.writerow([n, i + 1, mean[i], std[i], limit[i], abs(mean[i] - limit[i])])
                cells.append(f"{mean[i]:>9.5f} ±{std[i]:<7.5f}")
            print(f"{n:>6} " + " ".join(cells))
    print(f"{'limit':>6} " + " ".join(f"{v:>9.5f}{'':9}" for v in limit))


def plot_graphon(out, W):
    g = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(W.matrix(g), origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set(title=f"W(x,y) = {W.formula}", xlabel="x", ylabel="y")
    fig.tight_layout()
    fig.savefig(out / "graphon.png", dpi=150)
    plt.close(fig)


def plot_top(out, args, top, limit):
    ns = np.array(list(top))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(args.top_k):
        mean = [top[n][:, i].mean() for n in ns]
        std = [top[n][:, i].std() for n in ns]
        line = ax.errorbar(ns, mean, yerr=std, marker="o", capsize=3, label=f"λ{i + 1}/n")
        ax.axhline(limit[i], ls="--", lw=1, color=line[0].get_color())
    ax.set_xscale("log")
    ax.set(title="Top eigenvalues / n (dashed: graphon operator limit)", xlabel="n", ylabel="eigenvalue / n")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "top_eigenvalues.png", dpi=150)
    plt.close(fig)


def plot_spectrum(out, W, n, eigs):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(eigs / np.sqrt(n), bins=120, density=True, color="steelblue")
    ax.set(
        title=f"All eigenvalues / √n, n = {n}, W = {W.formula}",
        xlabel="eigenvalue / √n  (bulk stays O(√n); top eigenvalues grow like n)",
        ylabel="density",
    )
    fig.tight_layout()
    fig.savefig(out / f"spectrum_n{n}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
