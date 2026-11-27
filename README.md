# Give Me Some Credit — CSE 2600 final project

Predicting whether a borrower falls 90 or more days behind on a payment within
two years, using the Kaggle *Give Me Some Credit* dataset.

## The number that shapes everything else

The dataset holds 150,000 borrowers. 6.68% of them default.

That means a "model" that predicts nobody ever defaults is **93.32% accurate**
and completely useless. It never flags a single person who actually defaults.

This is why most of the evaluation here measures things other than accuracy. On
this dataset, accuracy mostly tells you what fraction of people did not default,
which we already knew before building anything.

## What is in the repo

```
data/
  cs-training.csv       the raw Kaggle file
  train.csv, test.csv   cleaned, split 80/20
  clean_data.py         second-pass cleaning, run from the raw file
  cleaning_params.json  the fill values and caps, measured on the training half
notebook/
  eda_daniel.ipynb      first look at the data
  clean_data.ipynb      the cleaning that produced train.csv and test.csv
src/
  evaluate.py           turns predictions into a table of numbers and figures
  check_evaluate.py     tests for evaluate.py
outputs/                model predictions go here
figures/                figures land here, created the first time you run
docs/
  wiki-outline.md       plan for the written wiki, page by page
```

## Running the evaluation

Each model saves its predictions to a file. One command then turns every
prediction file into a single comparison table and a set of figures.

### 1. Save predictions

Write `outputs/<model>_preds.csv` with three columns:

| column | what it holds |
|---|---|
| `id` | the borrower's original row number from the Kaggle file |
| `y_test` | what actually happened: `1` if they defaulted, `0` if not |
| `y_prob` | the model's estimated probability of default, between 0 and 1 |

Whatever comes before `_preds.csv` becomes the model's name in the table and on
the figures, so `outputs/logreg_preds.csv` appears as "logreg".

Two things to get right:

**`y_prob` must be a probability, not a 0 or 1 guess.** Most of the figures need
the probability itself. In scikit-learn that is
`model.predict_proba(X_test)[:, 1]`, not `model.predict(X_test)`.

**The `id` column matters more than it looks.** It lets the evaluation confirm
that both models were scored on the same people. If one of us used
`train.csv`/`test.csv` and the other re-ran `clean_data.py`, we would end up
with different test sets — and comparing two models on different sets of people
still produces a perfectly normal-looking chart that means nothing. With `id`
present, the evaluation stops and says so instead.

### 2. Run it

```
python src/evaluate.py
```

It reads everything in `outputs/`, prints the table, and writes the figures to
`figures/`. With no prediction files there yet, it says so and stops.

To use a cutoff other than 50%:

```
python src/evaluate.py --threshold 0.1
```

## What each figure shows

**ROC curve** (`roc_curve.png`) — how well a model sorts risky borrowers above
safe ones. The dashed diagonal is what you would get from flipping a coin. The
higher the curve sits above it, the better the sorting. The AUC number in the
legend is the area underneath: 0.5 is a coin flip, 1.0 is perfect.

**Precision-recall curve** (`pr_curve.png`) — of the people a model flags, what
share actually default, traced across every possible number of people flagged.
The flat dashed line is 0.067, which is what flagging people at random gives
you. This curve is the more honest one on a dataset this lopsided, because it
ignores the enormous number of safe borrowers the model correctly leaves alone.

**Calibration** (`calibration.png`) — whether the model's probabilities mean
anything. Group borrowers by predicted probability, then check what fraction of
each group actually defaulted. If the model says 30% and 30% of them default,
the point sits on the diagonal. Points above the line mean the model is
underestimating risk; below means it is overestimating.

**Confusion matrix** (`confusion_<model>.png`) — the four possible outcomes at a
chosen cutoff, with the top row being everyone who did not default and the
bottom row everyone who did. The percentages are shares of each row, so the
bottom row tells you directly what fraction of real defaulters the model caught
and what fraction it missed.

**Decile lift** (`lift_<model>.png`) — sort everyone from riskiest to safest and
cut the list into ten equal groups. Each bar shows how many times more
defaulters that group contains than the population average. A bar of 5.0 in the
first group means the riskiest tenth has five times the default rate of the
dataset as a whole. The line at 1.0 is what you would get by picking at random,
so bars above it are where the model is earning its keep.

## Reading the numbers

**What to expect.** A tuned gradient boosting model lands around 0.86–0.87
ROC-AUC on this data. Logistic regression lands a little below it. If either
comes back near 0.99, something has leaked — most likely information from the
test set got used while fitting.

**Why the 50% cutoff looks strange.** Only 6.68% of borrowers default, so a
well-behaved model rarely assigns anyone a probability above 50%. At that cutoff
it flags almost nobody: it raises very few false alarms, but it also misses most
of the people who actually default. That is not a bug, and it is the reason
`--threshold` exists. Lowering the cutoff catches more defaulters at the cost of
flagging more people who would have been fine, and choosing where to sit on that
trade-off is a decision about cost, not about accuracy.

## Tests

```
python src/check_evaluate.py
```

Six checks, all on generated data. They confirm that random predictions score
around 0.5, that perfect predictions score 1.0 with the highest lift in the
first group, that the error rates match a confusion matrix worked out by hand,
that prediction files survive a round trip to disk, that plain Python lists and
numpy arrays give identical answers, and that every figure function runs. The
generated data never touches `figures/`.
