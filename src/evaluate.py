"""
evaluate.py - the final results: the FROZEN model, calibration and decision rule
applied to the internal test set and to the two external hospitals.

Nothing here is fitted. Every parameter is read from files that were frozen and
pushed to GitHub BEFORE Shenzhen and Montgomery were scored:
    results/decision_rule.json   (Platt a, b + conformal thresholds)

Inputs : results/{internal_test,shenzhen,montgomery}_logits.csv
Outputs: results/final_results.csv           - the main results table
         results/score_distributions.png      - how the model scores TB vs normal per site
         results/reliability_<site>.png       - calibration at each site

Run from the project folder:   python src/evaluate.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

from calibrate import brier, ece, reliability_diagram, sigmoid
from selective import decide

RESULTS = "results"
SITES = ["internal_test", "shenzhen", "montgomery"]
N_BOOT, SEED = 1000, 42


def bootstrap_auroc(y, s, n_boot=N_BOOT, seed=SEED):
    """95% CI for AUROC: resample images with replacement, recompute, take 2.5/97.5 percentiles."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():             # need both classes in the resample
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def exact_ci(k, n):
    """Clopper-Pearson 95% CI for a proportion k/n (works even when k = 0)."""
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def evaluate_site(name, rule):
    d = pd.read_csv(os.path.join(RESULTS, f"{name}_logits.csv"))
    y, z = d["label"].to_numpy(), d["logit"].to_numpy()
    p = sigmoid(rule["platt"]["a"] * z + rule["platt"]["b"])       # frozen calibration
    dec = decide(p, rule)                                            # frozen decision rule
    tb, nt = y == 1, y == 0

    auc = roc_auc_score(y, z)
    auc_lo, auc_hi = bootstrap_auroc(y, z)
    missed = int(((dec == "TB unlikely") & tb).sum())
    miss_lo, miss_hi = exact_ci(missed, int(tb.sum()))
    cleared = int(((dec == "TB unlikely") & nt).sum())

    row = {
        "site": name,
        "images": len(y),
        "TB": int(tb.sum()),
        "AUROC": auc, "AUROC_CI_low": auc_lo, "AUROC_CI_high": auc_hi,
        "sensitivity_refer_counts_as_caught": 1 - missed / tb.sum(),
        "missed_TB_n": missed,
        "missed_TB_rate": missed / tb.sum(), "missed_CI_low": miss_lo, "missed_CI_high": miss_hi,
        "guarantee_held (missed <= 5%)": missed / tb.sum() <= 0.05,
        "specificity_healthy_cleared": cleared / nt.sum(),
        "false_alarms_n": int(((dec == "TB likely") & nt).sum()),
        "referral_rate": float((dec == "refer").mean()),
        "ECE": ece(p, y), "Brier": brier(p, y),
        "avg_predicted_TB": float(p.mean()), "real_TB_rate": float(y.mean()),
    }

    cm = pd.crosstab(pd.Series(np.where(tb, "really TB", "really not TB"), name="truth"),
                     pd.Series(dec, name="decision"))
    cm = cm.reindex(index=["really TB", "really not TB"],
                    columns=["TB likely", "refer", "TB unlikely"], fill_value=0)

    reliability_diagram(p, y, f"{name} - frozen Platt calibration",
                        os.path.join(RESULTS, f"reliability_{name}.png"))
    return row, cm, z, y


def plot_distributions(scores, path, rule):
    """Raw score histograms: if TB and normal overlap, the model cannot separate them there."""
    fig, axes = plt.subplots(1, len(scores), figsize=(15, 4), sharex=True)
    bins = np.linspace(-10, 14, 49)
    a, b = rule["platt"]["a"], rule["platt"]["b"]
    to_logit = lambda q: (np.log(q / (1 - q)) - b) / a          # thresholds back on the raw-score axis
    for ax, (name, (z, y)) in zip(axes, scores.items()):
        ax.hist(z[y == 0], bins=bins, alpha=0.6, density=True, label="really not TB")
        ax.hist(z[y == 1], bins=bins, alpha=0.6, density=True, label="really TB")
        for q in [rule["not_in_set_if_p_at_most"], rule["tb_in_set_if_p_at_least"]]:
            ax.axvline(to_logit(q), color="k", ls="--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("model score (logit)")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.suptitle("Score distributions per site (dashed lines = frozen decision thresholds)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    with open(os.path.join(RESULTS, "decision_rule.json")) as f:
        rule = json.load(f)

    rows, scores = [], {}
    for site in SITES:
        row, cm, z, y = evaluate_site(site, rule)
        rows.append(row); scores[site] = (z, y)
        print(f"\n===== {site} ({row['images']} images, {row['TB']} TB) =====")
        print(f"  AUROC              {row['AUROC']:.3f}  (95% CI {row['AUROC_CI_low']:.3f}-{row['AUROC_CI_high']:.3f})")
        print(f"  missed TB          {row['missed_TB_n']}/{row['TB']} = {100*row['missed_TB_rate']:.1f}%"
              f"  (95% CI {100*row['missed_CI_low']:.1f}-{100*row['missed_CI_high']:.1f}%)"
              f"  -> guarantee {'HELD' if row['guarantee_held (missed <= 5%)'] else 'BROKEN'}")
        print(f"  healthy cleared    {100*row['specificity_healthy_cleared']:.1f}%   false alarms {row['false_alarms_n']}")
        print(f"  referred to human  {100*row['referral_rate']:.1f}%")
        print(f"  ECE {row['ECE']:.3f} | Brier {row['Brier']:.3f} | avg predicted TB {row['avg_predicted_TB']:.3f}"
              f" vs real {row['real_TB_rate']:.3f}")
        print(cm.to_string())

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "final_results.csv"), index=False)
    plot_distributions(scores, os.path.join(RESULTS, "score_distributions.png"), rule)
    print("\nSaved: results/final_results.csv, results/score_distributions.png, results/reliability_<site>.png")
