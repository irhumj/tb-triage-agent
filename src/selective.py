"""
selective.py - the decision rule: TB likely / refer to a human / TB unlikely.

Method: class-conditional ("Mondrian") split conformal prediction.
  For each X-ray we build a SET of plausible answers:
      {TB}            -> "TB likely"     (send for confirmatory testing)
      {not TB}        -> "TB unlikely"   (cleared)
      {TB, not TB}    -> "refer"         (both plausible: a human must read it)
      {} (empty)      -> "refer"         (neither fits well: unusual image)

  We choose an error rate per class (ALPHA). Conformal prediction then sets the
  thresholds on the VAL set so that, IF new patients resemble the val patients:
      - at most ~ALPHA_TB of real TB cases are cleared as "TB unlikely"
      - at most ~ALPHA_NOT of real non-TB cases are excluded from "not TB"

  Design choice (option A): ALPHA_TB = ALPHA_NOT = 0.05.
  Reason: exceeds WHO's ~90% sensitivity target for triage, is statistically
  stable with 200 TB calibration cases, and keeps referrals and false alarms low.

Inputs : results/val_logits.csv, results/internal_test_logits.csv,
         results/calibration.json (Platt a, b - frozen earlier)
Outputs: results/decision_rule.json (FROZEN thresholds), results/risk_coverage.png

Known limitation: Platt scaling and the conformal thresholds are both fitted on
the same val set. Platt has only 2 parameters, so the effect is small, but the
guarantee is then approximate rather than exact.

Run from the project folder:   python src/selective.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = "results"
ALPHA_TB = 0.05     # max fraction of real TB cases we accept to clear wrongly
ALPHA_NOT = 0.05    # max fraction of real non-TB cases we accept to exclude from "not TB"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def conformal_quantile(scores, alpha):
    """The conformal threshold: a slightly conservative (1 - alpha) quantile.

    The (n + 1) correction is what gives the finite-sample guarantee.
    """
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def fit_rule(p, y):
    """Fit both thresholds on calibrated val probabilities p with labels y.

    Nonconformity score = 1 - (probability given to the TRUE class):
      for TB cases:     1 - p        (low p   = TB case looks unlike TB)
      for non-TB cases: p            (high p  = healthy case looks like TB)
    """
    q_tb = conformal_quantile(1 - p[y == 1], ALPHA_TB)
    q_not = conformal_quantile(p[y == 0], ALPHA_NOT)
    return {
        "tb_in_set_if_p_at_least": 1 - q_tb,     # TB stays plausible above this
        "not_in_set_if_p_at_most": q_not,        # "not TB" stays plausible below this
    }


def decide(p, rule):
    """Turn calibrated probabilities into the three decisions."""
    tb_in = p >= rule["tb_in_set_if_p_at_least"]
    not_in = p <= rule["not_in_set_if_p_at_most"]
    return np.where(tb_in & ~not_in, "TB likely",
           np.where(not_in & ~tb_in, "TB unlikely", "refer"))


def report(name, p, y, rule):
    d = decide(p, rule)
    tb, nt = y == 1, y == 0
    rows = {
        "images": len(y),
        "TB cases": int(tb.sum()),
        "referred to human (%)": 100 * (d == "refer").mean(),
        "TB cleared wrongly = MISSED (n)": int(((d == "TB unlikely") & tb).sum()),
        "missed TB rate (%)  [target <= 5]": 100 * (d[tb] == "TB unlikely").mean(),
        "sensitivity, refer counts as caught (%)": 100 * (d[tb] != "TB unlikely").mean(),
        "healthy cleared correctly (%)": 100 * (d[nt] == "TB unlikely").mean(),
        "false alarms: healthy called 'TB likely' (n)": int(((d == "TB likely") & nt).sum()),
    }
    print(f"\n--- {name} ---")
    for k, v in rows.items():
        print(f"  {k:46s} {v:.1f}" if isinstance(v, float) else f"  {k:46s} {v}")

    # 2 x 3 confusion matrix: truth (rows) x decision (columns)
    cm = pd.crosstab(pd.Series(np.where(tb, "really TB", "really not TB"), name="truth"),
                     pd.Series(d, name="decision"))
    cm = cm.reindex(index=["really TB", "really not TB"],
                    columns=["TB likely", "refer", "TB unlikely"], fill_value=0)
    print(cm.to_string())
    return d


def risk_coverage(p, y, path):
    """Error rate of the plain 50% rule if we only answer the most confident cases."""
    conf = np.maximum(p, 1 - p)
    wrong = ((p >= 0.5).astype(int) != y)
    order = np.argsort(-conf)
    n = len(y)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(wrong[order]) / np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(coverage, 100 * risk)
    ax.set_xlabel("Coverage: fraction of X-rays the model answers itself")
    ax.set_ylabel("Error rate on those X-rays (%)")
    ax.set_title("Risk-coverage curve (val): abstaining on the least confident")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    with open(os.path.join(RESULTS, "calibration.json")) as f:
        platt = json.load(f)["platt"]
    a, b = platt["a"], platt["b"]

    val = pd.read_csv(os.path.join(RESULTS, "val_logits.csv"))
    test = pd.read_csv(os.path.join(RESULTS, "internal_test_logits.csv"))
    p_val, y_val = sigmoid(a * val["logit"].to_numpy() + b), val["label"].to_numpy()
    p_test, y_test = sigmoid(a * test["logit"].to_numpy() + b), test["label"].to_numpy()

    rule = fit_rule(p_val, y_val)
    print("=== Decision rule fitted on VAL (Platt-calibrated probabilities) ===")
    print(f"  ALPHA_TB = {ALPHA_TB}, ALPHA_NOT = {ALPHA_NOT}")
    print(f"  TB stays plausible if p >= {rule['tb_in_set_if_p_at_least']:.4f}")
    print(f"  'not TB' stays plausible if p <= {rule['not_in_set_if_p_at_most']:.4f}")

    report("VAL (thresholds fitted here - optimistic)", p_val, y_val, rule)
    report("INTERNAL TEST (honest check, same hospitals)", p_test, y_test, rule)

    risk_coverage(p_val, y_val, os.path.join(RESULTS, "risk_coverage.png"))

    frozen = {"method": "class-conditional split conformal on Platt-calibrated probabilities",
              "alpha_tb": ALPHA_TB, "alpha_not": ALPHA_NOT,
              "platt": {"a": a, "b": b}, **rule, "fitted_on": "val"}
    with open(os.path.join(RESULTS, "decision_rule.json"), "w") as f:
        json.dump(frozen, f, indent=2)
    print("\nSaved (FROZEN): results/decision_rule.json and results/risk_coverage.png")
