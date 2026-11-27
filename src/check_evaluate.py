"""
check_evaluate.py  -  checks for evaluate.py, on synthetic data only.

What it checks (in order):
  1. Random scores give a ROC-AUC near 0.5.
  2. Perfect scores give a ROC-AUC of 1.0 and the highest lift in decile 1.
  3. FPR, FNR, accuracy and the misclassification rate match a confusion
     matrix worked out by hand.
  4. load_preds round-trips a small csv, and reports a missing id column.
  5. The functions give the same answer for plain lists and numpy arrays.
  6. Every plot function runs and writes its png.

All data here is generated. Figures from these checks go to a temporary
folder, so nothing synthetic can end up in figures/.

Usage:
    python src/check_evaluate.py

Needs: pandas, numpy, matplotlib, scikit-learn.
"""
import os
import tempfile

import matplotlib

# Headless backend, set before evaluate imports pyplot
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import evaluate

SEED = 2600
TOL = 1e-9
AUC_TOL = 0.02


def check_random_scores():
    rng = np.random.default_rng(SEED)
    n = 50000
    y_true = (rng.random(n) < 0.07).astype(int)
    y_prob = rng.random(n)
    auc = evaluate.model_metrics(y_true, y_prob)["ROC-AUC"]
    assert abs(auc - 0.5) < AUC_TOL, f"random scores gave ROC-AUC {auc:.4f}"
    print(f"1. random scores      ROC-AUC {auc:.4f}")


def check_perfect_scores():
    y_true = np.array([1] * 200 + [0] * 1800)
    y_prob = y_true.astype(float)
    m = evaluate.model_metrics(y_true, y_prob)
    assert m["ROC-AUC"] == 1.0, f"perfect scores gave ROC-AUC {m['ROC-AUC']}"
    assert m["PR-AUC"] == 1.0, f"perfect scores gave PR-AUC {m['PR-AUC']}"
    assert m["Misclassification"] == 0.0, f"misclassification {m['Misclassification']}"

    lift = evaluate.lift_by_decile(y_true, y_prob)
    assert lift.argmax() == 0, f"highest lift was in decile {lift.argmax() + 1}"
    assert lift[0] > 1.0, f"decile 1 lift was only {lift[0]}"
    print(f"2. perfect scores     ROC-AUC {m['ROC-AUC']:.4f}, decile 1 lift {lift[0]:.2f}")


def check_confusion_definitions():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.2, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1]
    # By hand at threshold 0.5: tp 2, fn 1, fp 2, tn 5, total 10
    counts = evaluate.confusion_counts(y_true, y_prob)
    assert counts == (5, 2, 1, 2), f"expected (tn, fp, fn, tp) = (5, 2, 1, 2), got {counts}"

    m = evaluate.model_metrics(y_true, y_prob)
    assert abs(m["FPR"] - 2 / 7) < TOL, f"FPR should be 2/7, got {m['FPR']}"
    assert abs(m["FNR"] - 1 / 3) < TOL, f"FNR should be 1/3, got {m['FNR']}"
    assert abs(m["Misclassification"] - 0.3) < TOL, f"got {m['Misclassification']}"
    assert abs(m["Accuracy"] - 0.7) < TOL, f"got {m['Accuracy']}"

    # A higher threshold must turn the two false positives into true negatives
    assert evaluate.confusion_counts(y_true, y_prob, 0.75) == (7, 0, 1, 2)

    # Each row must account for a whole actual class, which is what the
    # confusion plot divides by when it shades
    tn, fp, fn, tp = counts
    assert tn + fp == y_true.count(0), "row 0 should hold every actual negative"
    assert fn + tp == y_true.count(1), "row 1 should hold every actual positive"
    print(f"3. hand-checked cm    FPR {m['FPR']:.4f}, FNR {m['FNR']:.4f}, "
          f"misclassification {m['Misclassification']:.4f}")


def check_load_preds_round_trip():
    ids = [10, 20, 30, 40, 50]
    y_true = [0, 1, 0, 1, 1]
    y_prob = [0.11, 0.92, 0.34, 0.55, 0.78]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "roundtrip_preds.csv")
        pd.DataFrame({
            evaluate.ID_COL: ids,
            evaluate.TRUE_COL: y_true,
            evaluate.PROB_COL: y_prob,
        }).to_csv(path, index=False)

        got_true, got_prob = evaluate.load_preds(path)
        assert list(got_true) == y_true, f"y_true came back as {list(got_true)}"
        assert np.allclose(got_prob, y_prob), f"y_prob came back as {list(got_prob)}"
        assert list(evaluate.load_pred_ids(path)) == ids
        assert evaluate.model_name_from_path(path) == "roundtrip"

        # A file with no id column loads, but reports that it has no ids
        bare = os.path.join(tmp, "bare_preds.csv")
        pd.DataFrame({evaluate.TRUE_COL: y_true, evaluate.PROB_COL: y_prob}).to_csv(
            bare, index=False
        )
        assert evaluate.load_pred_ids(bare) is None

        # Probabilities written into the label column must be rejected, not scored
        swapped = os.path.join(tmp, "swapped_preds.csv")
        pd.DataFrame({evaluate.TRUE_COL: y_prob, evaluate.PROB_COL: y_prob}).to_csv(
            swapped, index=False
        )
        try:
            evaluate.load_preds(swapped)
        except ValueError:
            pass
        else:
            raise AssertionError("load_preds accepted probabilities as labels")

        # A file missing y_prob must fail loudly rather than guess
        wrong = os.path.join(tmp, "wrong_preds.csv")
        pd.DataFrame({evaluate.TRUE_COL: y_true}).to_csv(wrong, index=False)
        try:
            evaluate.load_preds(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("load_preds accepted a file with no y_prob column")
    print("4. load_preds         round-tripped 5 rows, id and error cases handled")


def check_lists_and_arrays():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    y_prob = [0.1, 0.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.05, 0.4, 0.55]
    from_lists = evaluate.model_metrics(y_true, y_prob)
    from_arrays = evaluate.model_metrics(np.array(y_true), np.array(y_prob))
    for key in from_lists:
        assert abs(from_lists[key] - from_arrays[key]) < TOL, f"{key} differs"

    table = evaluate.metrics_table({
        "lists": (y_true, y_prob),
        "arrays": (np.array(y_true), np.array(y_prob)),
    })
    assert list(table.index) == ["lists", "arrays"], f"index is {list(table.index)}"
    assert np.allclose(table.loc["lists"].to_numpy(), table.loc["arrays"].to_numpy())

    lift_lists = evaluate.lift_by_decile(y_true, y_prob, n_deciles=5)
    lift_arrays = evaluate.lift_by_decile(np.array(y_true), np.array(y_prob), n_deciles=5)
    assert np.allclose(lift_lists, lift_arrays)

    # Mismatched lengths must be caught, not broadcast
    try:
        evaluate.model_metrics([0, 1], [0.5])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched lengths were accepted")

    # One class only means ROC-AUC is undefined, so it must not be computed
    try:
        evaluate.model_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    except ValueError:
        pass
    else:
        raise AssertionError("a single-class y_true was accepted")
    print(f"5. lists vs arrays    {len(from_lists)} metrics agree to {TOL:g}")


def check_plots_run():
    rng = np.random.default_rng(SEED)
    n = 2000
    y_true = (rng.random(n) < 0.07).astype(int)
    y_prob = np.clip(0.04 + 0.45 * y_true + rng.normal(0, 0.12, n), 0.001, 0.999)
    results = {"model_a": (y_true, y_prob), "model_b": (y_true, 1 - y_prob * 0.5)}

    with tempfile.TemporaryDirectory() as tmp:
        evaluate.plot_roc(results, tmp)
        evaluate.plot_pr(results, tmp)
        evaluate.plot_calibration(results, outdir=tmp)
        evaluate.plot_confusion(y_true, y_prob, 0.5, "model_a", tmp)
        evaluate.plot_lift(y_true, y_prob, "model_a", outdir=tmp)
        written = sorted(os.listdir(tmp))
        assert len(written) == 5, f"expected 5 png files, got {written}"

    leaked = []
    if os.path.isdir(evaluate.FIGDIR):
        leaked = [f for f in os.listdir(evaluate.FIGDIR) if "model_a" in f or "model_b" in f]
    assert not leaked, f"synthetic figures leaked into {evaluate.FIGDIR}/: {leaked}"
    print(f"6. plots              wrote {len(written)} png files, none in {evaluate.FIGDIR}/")


def main():
    check_random_scores()
    check_perfect_scores()
    check_confusion_definitions()
    check_load_preds_round_trip()
    check_lists_and_arrays()
    check_plots_run()
    print("All checks passed.")


if __name__ == "__main__":
    main()
