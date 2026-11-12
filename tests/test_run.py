"""End-to-end: run.py produces all its output files with consistent contents."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def run(out, *args):
    cmd = [sys.executable, str(ROOT / "run.py"), "--out", str(out), *args]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_outputs(tmp_path):
    stdout = run(tmp_path, "--formula", "x*y", "--n", "30", "50", "--trials", "3", "--top-k", "2", "--draw-n", "20")
    assert "0.33333" in stdout

    names = {p.name for p in tmp_path.iterdir()}
    assert names == {
        "eigenvalues.csv", "summary.csv", "multiplicities_n30.txt", "multiplicities_n50.txt",
        "graphon.png", "top_eigenvalues.png", "spectrum_n50.png", "graph.png",
    }

    eig_rows = read_csv(tmp_path / "eigenvalues.csv")
    assert len(eig_rows) == 3 * 30 + 3 * 50
    for row in eig_rows:
        assert float(row["eigenvalue_over_n"]) == pytest.approx(float(row["eigenvalue"]) / int(row["n"]))
    # each sampled graph's eigenvalues sum to zero (trace of A)
    for n in (30, 50):
        for t in range(3):
            total = sum(float(r["eigenvalue"]) for r in eig_rows if r["n"] == str(n) and r["trial"] == str(t))
            assert total == pytest.approx(0, abs=1e-8)

    summary = read_csv(tmp_path / "summary.csv")
    assert len(summary) == 2 * 2
    assert float(summary[0]["graphon_limit"]) == pytest.approx(1 / 3, abs=1e-6)

    # summary means agree with the raw eigenvalues
    lam1_n30 = [float(r["eigenvalue_over_n"]) for r in eig_rows if r["n"] == "30" and r["index"] == "0"]
    assert float(summary[0]["mean_eig_over_n"]) == pytest.approx(np.mean(lam1_n30))

    text = (tmp_path / "multiplicities_n30.txt").read_text()
    counts = [int(line.split()[-1]) for line in text.splitlines()[4:] if line.strip()]
    assert sum(counts) == 30


def test_same_seed_reproducible(tmp_path):
    args = ("--formula", "exp(-abs(x-y))", "--n", "40", "--trials", "2", "--seed", "9")
    run(tmp_path / "a", *args)
    run(tmp_path / "b", *args)
    assert (tmp_path / "a" / "eigenvalues.csv").read_text() == (tmp_path / "b" / "eigenvalues.csv").read_text()


def test_invalid_formula_fails(tmp_path):
    cmd = [sys.executable, str(ROOT / "run.py"), "--out", str(tmp_path), "--formula", "x"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "not symmetric" in result.stderr
