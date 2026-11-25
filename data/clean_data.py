"""
clean_data.py  -  second-pass cleaning for the Give Me Some Credit dataset.

What it does (in order):
  1. Loads the raw Kaggle file (cs-training.csv).
  2. Drops rows that are impossible (age under 18).
  3. Adds flag columns BEFORE anything is filled in, so no information is lost:
       income_missing   1 if MonthlyIncome was NA
       late_code_flag   1 if any late-payment count was the placeholder 96 or 98
  4. Splits 80/20 into train and test, stratified on SeriousDlqin2yrs.
  5. Computes every fill value and cap from the TRAIN set only, then applies
     the same values to train and test (no test-set leakage).
  6. Fills / fixes:
       MonthlyIncome            NA -> train median of reported incomes
       NumberOfDependents       NA -> train median
       late-payment counts      96/98 -> train median of the normal values (flag keeps the signal)
       DebtRatio                rows with missing income have a fake ratio (raw debt, not
                                debt / income), so it is replaced with the train median of
                                real ratios (income_missing keeps the signal)
       caps (99th percentile)   RevolvingUtilizationOfUnsecuredLines, DebtRatio, MonthlyIncome
  7. Checks the result and writes train_clean.csv, test_clean.csv, cleaning_params.json.

Usage:
    python clean_data.py                                   # looks for cs-training.csv in the current folder
    python clean_data.py --input path/to/cs-training.csv --outdir data/clean
    python clean_data.py --seed 42                         # change the random split

Needs: pandas, scikit-learn.
"""
import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "SeriousDlqin2yrs"
LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
PLACEHOLDERS = [96, 98]
CAP_COLS = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
CAP_QUANTILE = 0.99


def load_raw(path):
    df = pd.read_csv(path, na_values=["NA"])
    df = df.rename(columns={"Unnamed: 0": "id"})
    return df


def add_flags_and_drop_invalid(df):
    n0 = len(df)
    df = df[df["age"] >= 18].copy()
    print(f"Dropped {n0 - len(df)} row(s) with age < 18")

    df["income_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["late_code_flag"] = df[LATE_COLS].isin(PLACEHOLDERS).any(axis=1).astype(int)
    print(f"income_missing = 1 on {df['income_missing'].sum()} rows")
    print(f"late_code_flag = 1 on {df['late_code_flag'].sum()} rows")
    return df


def fit_params(train):
    """Everything data-driven is computed here, from the training set only."""
    reported = train[train["income_missing"] == 0]
    params = {
        "income_median": float(reported["MonthlyIncome"].median()),
        "dependents_median": float(train["NumberOfDependents"].median()),
        "debtratio_median_reported_income": float(reported["DebtRatio"].median()),
        "late_medians": {
            c: float(train.loc[~train[c].isin(PLACEHOLDERS), c].median()) for c in LATE_COLS
        },
        "caps": {},
    }
    # Caps use rows with real (reported-income) DebtRatio / income values
    params["caps"]["RevolvingUtilizationOfUnsecuredLines"] = float(
        train["RevolvingUtilizationOfUnsecuredLines"].quantile(CAP_QUANTILE)
    )
    params["caps"]["DebtRatio"] = float(reported["DebtRatio"].quantile(CAP_QUANTILE))
    params["caps"]["MonthlyIncome"] = float(reported["MonthlyIncome"].quantile(CAP_QUANTILE))
    return params


def apply_params(df, p):
    df = df.copy()

    # 96/98 placeholders -> normal value (flag column already records them)
    for c in LATE_COLS:
        df.loc[df[c].isin(PLACEHOLDERS), c] = p["late_medians"][c]

    # Fake DebtRatio on missing-income rows -> median of real ratios
    df.loc[df["income_missing"] == 1, "DebtRatio"] = p["debtratio_median_reported_income"]

    # Fill NAs
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(p["income_median"])
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(p["dependents_median"])

    # Cap extreme values at the train 99th percentile
    for c, cap in p["caps"].items():
        df[c] = df[c].clip(upper=cap)
    return df


def check(train, test, params):
    """Fail loudly if any fix did not take effect."""
    for name, d in [("train", train), ("test", test)]:
        assert d.drop(columns=[]).isna().sum().sum() == 0, f"{name}: still has NAs"
        assert (d["age"] >= 18).all(), f"{name}: age < 18 remains"
        assert not d[LATE_COLS].isin(PLACEHOLDERS).any().any(), f"{name}: 96/98 remains"
        for c, cap in params["caps"].items():
            assert d[c].max() <= cap + 1e-9, f"{name}: {c} exceeds cap"
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.002, "split not stratified"
    print("All checks passed.")


def summarize(name, d):
    print(
        f"{name}: {len(d):,} rows | default rate {d[TARGET].mean():.4f} | "
        f"income_missing {d['income_missing'].sum():,} | late_code_flag {d['late_code_flag'].sum():,}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="cs-training.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--seed", type=int, default=2600)
    args = ap.parse_args()

    df = load_raw(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")
    df = add_flags_and_drop_invalid(df)

    # Split FIRST, then compute everything from train
    train, test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=args.seed
    )
    params = fit_params(train)
    train_c = apply_params(train, params)
    test_c = apply_params(test, params)

    check(train_c, test_c, params)
    summarize("train", train_c)
    summarize("test ", test_c)

    os.makedirs(args.outdir, exist_ok=True)
    train_c.sort_values("id").to_csv(os.path.join(args.outdir, "train_clean.csv"), index=False)
    test_c.sort_values("id").to_csv(os.path.join(args.outdir, "test_clean.csv"), index=False)
    with open(os.path.join(args.outdir, "cleaning_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"Wrote train_clean.csv, test_clean.csv, cleaning_params.json to {args.outdir}")


if __name__ == "__main__":
    main()
