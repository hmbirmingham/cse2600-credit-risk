"""
evaluate.py  -  model evaluation and plots for the Give Me Some Credit project.

What it does (in order):
  1. Finds every predictions file in the outputs folder (<model>_preds.csv).
  2. Checks that the models were all scored on the same test rows, when the
     files carry an id column, and stops if they were not.
  3. Builds one metrics table: ROC-AUC, PR-AUC, accuracy, misclassification
     rate, FPR and FNR at the decision threshold.
  4. Draws and saves the shared plots (ROC, precision-recall, calibration) and
     the per-model plots (confusion matrix, decile lift).

Every metric function takes y_true and y_prob, where y_prob is the predicted
probability of the positive class (SeriousDlqin2yrs == 1). Lists and numpy
arrays both work. The threshold defaults to 0.5 and is always a parameter.

Definitions, to match the ones used in class:
  FPR                FP / (FP + TN)
  FNR                FN / (FN + TP)
  misclassification  (FP + FN) / total

Predictions file format:
    outputs/<model>_preds.csv with columns id, y_test, y_prob
    id is optional, but without it a split mismatch cannot be detected.

Usage:
    python src/evaluate.py                                  # outputs/ -> figures/
    python src/evaluate.py --preddir outputs --figdir figures
    python src/evaluate.py --threshold 0.1                  # move the cutoff

Needs: pandas, numpy, matplotlib, scikit-learn.
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

THRESHOLD = 0.5
N_DECILES = 10
N_CAL_BINS = 10
FIGSIZE = (8, 6)
FIGSIZE_SQUARE = (6, 5.5)
DPI = 150
FIGDIR = "figures"
PREDDIR = "outputs"
PRED_GLOB = "*_preds.csv"
ID_COL = "id"
TRUE_COL = "y_test"
PROB_COL = "y_prob"


def as_arrays(y_true, y_prob):
    """Accept lists or arrays, return 1-D arrays of the same length."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true has {len(y_true)} rows but y_prob has {len(y_prob)}")
    if len(y_true) == 0:
        raise ValueError("no rows to evaluate")

    # A wrong column here would otherwise produce a plausible but meaningless chart
    labels = set(np.unique(y_true).tolist())
    if not labels <= {0, 1}:
        raise ValueError(
            f"{TRUE_COL} must hold 0/1 labels, found {sorted(labels)[:5]} - "
            "is it holding probabilities by mistake?"
        )
    if len(labels) == 1:
        raise ValueError(
            f"{TRUE_COL} is all {labels.pop():g} - ROC-AUC and lift need both classes"
        )
    return y_true.astype(int), y_prob


def load_preds(path):
    df = pd.read_csv(path)
    for c in (TRUE_COL, PROB_COL):
        if c not in df.columns:
            raise ValueError(f"{path}: no '{c}' column (found {list(df.columns)})")
    return as_arrays(df[TRUE_COL], df[PROB_COL])


def load_pred_ids(path):
    """Row ids from a predictions file, or None when it has no id column."""
    df = pd.read_csv(path)
    if ID_COL not in df.columns:
        return None
    return df[ID_COL].to_numpy()


def model_name_from_path(path):
    return os.path.basename(path).replace("_preds.csv", "")


def confusion_counts(y_true, y_prob, threshold=THRESHOLD):
    """Returns tn, fp, fn, tp at the given threshold."""
    y_true, y_prob = as_arrays(y_true, y_prob)
    pred = (y_prob >= threshold).astype(int)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tp = int(((pred == 1) & (y_true == 1)).sum())
    return tn, fp, fn, tp


def model_metrics(y_true, y_prob, threshold=THRESHOLD):
    y_true, y_prob = as_arrays(y_true, y_prob)
    tn, fp, fn, tp = confusion_counts(y_true, y_prob, threshold)
    total = tn + fp + fn + tp
    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Accuracy": (tp + tn) / total,
        "Misclassification": (fp + fn) / total,
        "FPR": fp / (fp + tn) if (fp + tn) else float("nan"),
        "FNR": fn / (fn + tp) if (fn + tp) else float("nan"),
    }


def metrics_table(results, threshold=THRESHOLD):
    rows = {name: model_metrics(yt, yp, threshold) for name, (yt, yp) in results.items()}
    return pd.DataFrame(rows).T


def lift_by_decile(y_true, y_prob, n_deciles=N_DECILES):
    """Decile success rate / overall success rate, highest predicted risk first."""
    y_true, y_prob = as_arrays(y_true, y_prob)
    overall = y_true.mean()
    if overall == 0:
        raise ValueError("no positive cases, lift is undefined")
    order = np.argsort(-y_prob)
    groups = np.array_split(y_true[order], n_deciles)
    return np.array([g.mean() / overall for g in groups])


def save_figure(fig, name, outdir=FIGDIR):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"Wrote {path}")
    return path


def plot_roc(results, outdir=FIGDIR):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, (yt, yp) in results.items():
        yt, yp = as_arrays(yt, yp)
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(yt, yp):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save_figure(fig, "roc_curve.png", outdir)
    return fig


def plot_pr(results, outdir=FIGDIR):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    baseline = None
    for name, (yt, yp) in results.items():
        yt, yp = as_arrays(yt, yp)
        if baseline is None:
            baseline = yt.mean()
        precision, recall, _ = precision_recall_curve(yt, yp)
        ax.plot(recall, precision, lw=2,
                label=f"{name} (PR-AUC = {average_precision_score(yt, yp):.3f})")
    # Baseline for a precision-recall curve is the positive rate, not 0.5
    ax.axhline(baseline, color="k", ls="--", lw=1,
               label=f"Baseline / positive rate ({baseline:.4f})")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    save_figure(fig, "pr_curve.png", outdir)
    return fig


def plot_confusion(y_true, y_prob, threshold=THRESHOLD, name="model", outdir=FIGDIR):
    tn, fp, fn, tp = confusion_counts(y_true, y_prob, threshold)
    matrix = np.array([[tn, fp], [fn, tp]])
    total = matrix.sum()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            text_color = "white" if count > matrix.max() / 2 else "black"
            ax.text(j, i, f"{count:,}\n{count / total:.2%}",
                    ha="center", va="center", color=text_color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title(f"{name}  -  confusion matrix at threshold {threshold:g}")
    save_figure(fig, f"confusion_{name}.png", outdir)
    return fig


def plot_lift(y_true, y_prob, name="model", n_deciles=N_DECILES, outdir=FIGDIR):
    lift = lift_by_decile(y_true, y_prob, n_deciles)
    deciles = np.arange(1, len(lift) + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(deciles, lift, color="steelblue")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="No better than random (lift = 1)")
    for d, value in zip(deciles, lift):
        ax.text(d, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(deciles)
    ax.set_xlabel("Decile of predicted risk (1 = highest risk)")
    ax.set_ylabel("Lift")
    ax.set_title(f"{name}  -  decile lift")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, f"lift_{name}.png", outdir)
    return fig


def plot_calibration(results, n_bins=N_CAL_BINS, outdir=FIGDIR):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    edges = np.linspace(0, 1, n_bins + 1)
    furthest = 0.0
    for name, (yt, yp) in results.items():
        yt, yp = as_arrays(yt, yp)
        which = np.clip(np.digitize(yp, edges) - 1, 0, n_bins - 1)
        predicted, observed = [], []
        for b in range(n_bins):
            rows = which == b
            if not rows.any():
                continue
            predicted.append(yp[rows].mean())
            observed.append(yt[rows].mean())
        furthest = max(furthest, max(predicted), max(observed))
        ax.plot(predicted, observed, marker="o", lw=2, label=name)

    # Most predictions sit near the 6.7% base rate, so zoom to where the data is
    limit = min(1.0, max(furthest * 1.1, 0.01))
    ax.plot([0, limit], [0, limit], "k--", lw=1, label="Perfectly calibrated")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration by bin")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    save_figure(fig, "calibration.png", outdir)
    return fig


def check_same_rows(ids_by_model):
    """Fail loudly if two models were scored on different test rows."""
    known = {n: i for n, i in ids_by_model.items() if i is not None}
    missing = [n for n, i in ids_by_model.items() if i is None]
    if missing:
        print(f"WARNING: no '{ID_COL}' column in {', '.join(missing)} - "
              "cannot confirm the models share a test set")
    if len(known) < 2:
        return
    names = list(known)
    first = names[0]
    for other in names[1:]:
        assert len(known[other]) == len(known[first]), (
            f"{other} has {len(known[other]):,} rows but {first} has {len(known[first]):,}"
        )
        assert set(known[other]) == set(known[first]), (
            f"{other} and {first} were scored on different test rows - "
            "the models did not use the same split"
        )
    print(f"Row ids match across {len(known)} model(s).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preddir", default=PREDDIR)
    ap.add_argument("--figdir", default=FIGDIR)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    pattern = os.path.join(args.preddir, PRED_GLOB)
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No prediction files found at {pattern} - nothing to evaluate yet.")
        print(f"Each model should write {args.preddir}/<model>_preds.csv "
              f"with columns {ID_COL}, {TRUE_COL}, {PROB_COL}.")
        return

    results, ids = {}, {}
    for path in paths:
        name = model_name_from_path(path)
        results[name] = load_preds(path)
        ids[name] = load_pred_ids(path)
        y_true, _ = results[name]
        print(f"Loaded {name}: {len(y_true):,} rows | "
              f"positive rate {y_true.mean():.4f} | {path}")
    check_same_rows(ids)

    print(f"\nMetrics at threshold {args.threshold:g}")
    print(metrics_table(results, args.threshold).round(4).to_string())

    print()
    plot_roc(results, args.figdir)
    plot_pr(results, args.figdir)
    plot_calibration(results, outdir=args.figdir)
    for name, (y_true, y_prob) in results.items():
        plot_confusion(y_true, y_prob, args.threshold, name, args.figdir)
        plot_lift(y_true, y_prob, name, outdir=args.figdir)
    print(f"\nEvaluated {len(results)} model(s).")


if __name__ == "__main__":
    main()
