"""
predict.py - score ONE chest X-ray with the frozen pipeline.

This is the single place where "the model decides". The MCP server, the agent and the
web app all call TBScreener.predict(); none of them recompute probabilities or thresholds.

Frozen inputs (never changed after external testing):
    results/best_model.pt        trained DenseNet121 weights
    results/decision_rule.json   Platt a, b + conformal thresholds
    results/final_results.csv    external-validation numbers, shown as a warning

Usage:
    python src/predict.py path/to/xray.png
"""

import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from data import IMG_SIZE, load_xray, normalize

RESULTS = "results"

NEXT_STEP = {
    "TB likely": "Refer for confirmatory TB testing (e.g. sputum test).",
    "refer": "The model is not confident. A human reader should review this X-ray.",
    "TB unlikely": "No TB signal found by the model at triage. Clinical judgement still applies.",
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def decide(p, rule):
    """Same three-way rule as selective.py, for one probability."""
    tb_in = p >= rule["tb_in_set_if_p_at_least"]
    not_in = p <= rule["not_in_set_if_p_at_most"]
    if tb_in and not not_in:
        return "TB likely"
    if not_in and not tb_in:
        return "TB unlikely"
    return "refer"


def validation_warning(results_dir=RESULTS):
    """Build the warning from the saved results, so it can never drift from the real numbers."""
    path = os.path.join(results_dir, "final_results.csv")
    if not os.path.exists(path):
        return "Research prototype, not for diagnosis. External validation results not found."
    t = pd.read_csv(path).set_index("site")
    return (
        "Research prototype, not for diagnosis. "
        f"AUROC was {t.loc['internal_test', 'AUROC']:.4f} on TBX11K images, but only "
        f"{t.loc['shenzhen', 'AUROC']:.2f} (Shenzhen) and {t.loc['montgomery', 'AUROC']:.2f} (Montgomery) "
        "on hospitals not seen in training. At Montgomery "
        f"{100 * t.loc['montgomery', 'missed_TB_rate']:.0f}% of TB cases were missed, so the 5% "
        "missed-TB guarantee does not hold outside TBX11K-like images."
    )


class TBScreener:
    def __init__(self, results_dir=RESULTS, device=None):
        self.results_dir = results_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(results_dir, "decision_rule.json")) as f:
            self.rule = json.load(f)

        model = models.densenet121(weights=None)                 # architecture only, no download
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        state = torch.load(os.path.join(results_dir, "best_model.pt"), map_location=self.device)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self.warning = validation_warning(results_dir)

    def _tensor(self, image_path):
        img = load_xray(image_path)                              # same loader as training
        if img.std() < 1e-3:
            raise ValueError("Image is blank or unreadable.")
        x = torch.from_numpy(img)[None]
        return img, normalize(x.repeat(3, 1, 1))

    def _gradcam(self, x):
        """Grad-CAM written out by hand: same steps as DenseNet's own forward pass."""
        x = x[None].to(self.device)
        feats = self.model.features(x)
        feats.retain_grad()
        logit = self.model.classifier(F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1)).squeeze()
        self.model.zero_grad()
        logit.backward()
        w = feats.grad[0].mean(dim=(1, 2))
        cam = F.relu((w[:, None, None] * feats[0]).sum(0))
        cam = F.interpolate(cam[None, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear")[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam.detach().cpu().numpy(), float(logit.item())

    def predict(self, image_path, gradcam_path=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        img, x = self._tensor(image_path)

        if gradcam_path:
            cam, z = self._gradcam(x)
            gray = (img * 255).astype(np.uint8)
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.6, heat, 0.4, 0)
            cv2.imwrite(gradcam_path, overlay)
        else:
            with torch.no_grad():
                z = float(self.model(x[None].to(self.device)).squeeze().item())

        p = float(sigmoid(self.rule["platt"]["a"] * z + self.rule["platt"]["b"]))   # frozen Platt
        decision = decide(p, self.rule)                                               # frozen rule
        return {
            "file": os.path.basename(image_path),
            "decision": decision,
            "tb_probability": round(p, 4),
            "raw_logit": round(z, 3),
            "thresholds": {
                "tb_unlikely_if_p_at_most": round(self.rule["not_in_set_if_p_at_most"], 4),
                "tb_likely_if_p_at_least": round(self.rule["tb_in_set_if_p_at_least"], 4),
            },
            "next_step": NEXT_STEP[decision],
            "gradcam_png": gradcam_path,
            "validation_warning": self.warning,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python src/predict.py path/to/xray.png")
    screener = TBScreener()
    out = screener.predict(sys.argv[1], gradcam_path="results/last_gradcam.png")
    print(json.dumps(out, indent=2))
