"""
calibrate.py - can we trust the model's percentages?

PART 1: measure calibration BEFORE any fix.
  - turn raw scores (logits) into probabilities
  - reliability diagram: "when the model says X%, how often is it really TB?"
  - ECE and Brier score: one-number summaries of how honest the percentages are

PART 2: fix it, and check the fix on data it was NOT fitted on.
  - temperature scaling: p = sigmoid(z / T)          (1 knob: stretch/squash)
  - Platt scaling:       p = sigmoid(a * z + b)      (2 knobs: stretch + shift)
  - both are fitted on VAL only, then checked on INTERNAL TEST
  - the chosen method's numbers are frozen in results/calibration.json

Shenzhen and Montgomery are NOT used here.
Run from the project folder:   python src/calibrate.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")                     # save figures to files, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

RESULTS = "results"


def sigmoid(z):
    """Raw score (logit) -> probability between 0 and 1."""
    return 1 / (1 + np.exp(-z))


def calibration_bins(p, y, n_bins):
    """Group predictions into equal-width bins (0-10%, 10-20%, ...).

    For each bin return: how many images, the average predicted probability
    ("confidence"), and the real fraction of TB ("what actually happened").
    """
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)   # bin number per image
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(m.sum()),
            "predicted": p[m].mean() if m.any() else np.nan,
            "actual_tb_rate": y[m].mean() if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def ece(p, y, n_bins=15):
    """Expected Calibration Error: average gap between predicted and actual,
    weighted by how many images fall in each bin. 0 = perfectly honest."""
    t = calibration_bins(p, y, n_bins).dropna()
    return float((t["n"] / len(p) * (t["predicted"] - t["actual_tb_rate"]).abs()).sum())


def brier(p, y):
    """Mean squared difference between probability and truth (0 or 1). Lower is better."""
    return float(np.mean((p - y) ** 2))


def reliability_diagram(p, y, title, path, n_bins=10):
    """Bar chart: actual TB rate per bin, against the diagonal of perfect honesty."""
    t = calibration_bins(p, y, n_bins)
    centers = np.linspace(0.05, 0.95, n_bins)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.bar(centers, t["actual_tb_rate"].fillna(0), width=0.09,
           edgecolor="black", alpha=0.75, label="actual TB rate in bin")
    ax.plot([0, 1], [0, 1], "k--", label="perfectly honest")
    for c, (n, a) in zip(centers, zip(t["n"], t["actual_tb_rate"].fillna(0))):
        ax.text(c, a + 0.02, f"n={n}", ha="center", fontsize=7)
    ax.set_xlabel("What the model said (predicted probability of TB)")
    ax.set_ylabel("What actually happened (fraction that were TB)")
    ax.set_title(f"{title}\nECE={ece(p, y):.4f}  Brier={brier(p, y):.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.08); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def nll(p, y):
    """Negative log-likelihood: the loss we minimise to fit the calibration knobs."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(z, y):
    """Find the single number T that makes sigmoid(z / T) most honest on these labels."""
    res = minimize(lambda v: nll(sigmoid(z / np.exp(v[0])), y), x0=[0.0], method="Nelder-Mead")
    return float(np.exp(res.x[0]))          # fitted in log-space so T stays positive


def fit_platt(z, y):
    """Find a (stretch) and b (shift) that make sigmoid(a * z + b) most honest."""
    res = minimize(lambda v: nll(sigmoid(v[0] * z + v[1]), y), x0=[1.0, 0.0], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def summary_row(name, p, y):
    return {"method": name, "ECE": ece(p, y), "Brier": brier(p, y), "NLL": nll(p, y),
            "avg_predicted": float(p.mean()), "real_TB_rate": float(y.mean())}


if __name__ == "__main__":
    val = pd.read_csv(os.path.join(RESULTS, "val_logits.csv"))
    test = pd.read_csv(os.path.join(RESULTS, "internal_test_logits.csv"))
    z_val, y_val = val["logit"].to_numpy(), val["label"].to_numpy()
    z_test, y_test = test["logit"].to_numpy(), test["label"].to_numpy()

    # ---------- PART 1: before ----------
    p_before = sigmoid(z_val)
    print("=== PART 1: VAL set, BEFORE calibration ===")
    print(f"Images: {len(y_val)} | real TB rate: {y_val.mean():.3f} | average predicted TB: {p_before.mean():.3f}")
    print(f"ECE (15 bins): {ece(p_before, y_val):.4f} | Brier: {brier(p_before, y_val):.4f}\n")
    print(calibration_bins(p_before, y_val, 10).round(3).to_string(index=False))
    reliability_diagram(p_before, y_val, "Val - before calibration",
                        os.path.join(RESULTS, "reliability_before.png"))

    # ---------- PART 2: fit both fixes on VAL only ----------
    T = fit_temperature(z_val, y_val)
    a, b = fit_platt(z_val, y_val)
    print("\n=== PART 2: fitted on VAL ===")
    print(f"Temperature scaling: T = {T:.3f}")
    print(f"Platt scaling:       a = {a:.3f}, b = {b:.3f}")
    print(f"(for comparison, log(pos_weight) = log(10.02) = {np.log(10.02):.3f})")

    methods = {
        "before (raw)": lambda z: sigmoid(z),
        "temperature":  lambda z: sigmoid(z / T),
        "platt":        lambda z: sigmoid(a * z + b),
    }

    # ---------- Honest check on INTERNAL TEST (not used for fitting) ----------
    for split, z, y in [("VAL (fitted here - optimistic)", z_val, y_val),
                        ("INTERNAL TEST (honest check)", z_test, y_test)]:
        table = pd.DataFrame([summary_row(m, f(z), y) for m, f in methods.items()])
        print(f"\n--- {split} ---")
        print(table.round(4).to_string(index=False))

    reliability_diagram(sigmoid(z_test / T), y_test, f"Internal test - temperature (T={T:.2f})",
                        os.path.join(RESULTS, "reliability_temperature.png"))
    reliability_diagram(sigmoid(a * z_test + b), y_test, f"Internal test - Platt (a={a:.2f}, b={b:.2f})",
                        os.path.join(RESULTS, "reliability_platt.png"))
    reliability_diagram(sigmoid(z_test), y_test, "Internal test - before calibration",
                        os.path.join(RESULTS, "reliability_test_before.png"))

    # ---------- Freeze the parameters ----------
    with open(os.path.join(RESULTS, "calibration.json"), "w") as f:
        json.dump({"temperature": {"T": T}, "platt": {"a": a, "b": b},
                   "fitted_on": "val", "n_val": int(len(y_val))}, f, indent=2)
    print("\nSaved: results/calibration.json and reliability_*.png figures")
