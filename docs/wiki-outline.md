# Wiki Outline — CSE 2600 Final Project

Planning document mapping the five required wiki pages (§3(a) of the project
guidelines) onto the Give Me Some Credit dataset.

**Dataset:** Kaggle *Give Me Some Credit* (2011). 150,000 labeled borrowers,
10 predictors, target `SeriousDlqin2yrs`. Data lives in `data/raw/` and is
git-ignored — the competition rules restrict redistribution, so it stays out
of the repo. Teammates download it themselves (see below).

**Guiding habit:** no claim without a number or worked example attached. The
guidelines warn on page 2 that vague AI-generated phrasing such as "complex
nonlinear relationships" will need significant editing to pass. Every
assertion below is tied to a measured figure.

---

## Page 1 — Introduction

**Problem statement.** Predict whether a borrower experiences serious
delinquency (90+ days past due) within two years, using financial attributes
known at decision time.

**Open with the asymmetry.** A model predicting "never defaults" for all
150,000 borrowers is **93.32% accurate and worthless**. Stating this up front
gives the rest of the wiki a reason to exist.

**Context.** Consumer credit scoring under asymmetric costs: a rejected good
borrower is lost margin, an approved defaulter is lost principal.

**Scope.** 150,000 labeled borrowers, 10 predictors, 6.68% base rate.

**Required:** table of contents with links (§3(a)(i)). Feeds the 20%
organization criterion.

---

## Page 2 — Data

Carries the most rubric weight: the 20% data/methods criterion plus the
believability appeal.

### Believability

Real anonymized consumer records; 924 competing teams. Ground it in
distributions a reader can sanity-check: median income $5,400/mo, median age
52, median revolving utilization 0.154.

### Feature summary (grouped, not a flat list)

| Group | Features |
|---|---|
| Leverage | `RevolvingUtilizationOfUnsecuredLines`, `DebtRatio`, `MonthlyIncome` |
| Delinquency history | 30–59, 60–89, and 90+ day past-due counters |
| Portfolio | `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines` |
| Demographic | `age`, `NumberOfDependents` |

### Pre-processing — four decisions, each with evidence

1. **Sentinel codes.** 269 rows carry 96/98, and it is the *same* 269 rows in
   all three counters; excluding them the columns cap at 13/17/11. Those rows
   default at **54.6%** vs 6.6% elsewhere. Decision: encode an indicator,
   do **not** drop — dropping discards the most predictive subgroup in the file.
2. **`MonthlyIncome`, 19.8% missing** (29,731 rows). Missing-income rows
   default *less* — 5.61% vs 6.95% — so missingness is informative, not
   missing-at-random. Decision: median-impute **plus** a missingness flag;
   show what plain mean imputation costs.
3. **`NumberOfDependents`, 2.6% missing.** Argue 0 vs median; state the choice.
4. **Scale violations.** A utilization *ratio* reaching 50,708; `DebtRatio`
   reaching 329,664; income to $3,008,750; one row with `age` = 0. Winsorize
   or log-transform. Note that trees are invariant to monotone transforms
   while logistic regression is not — this is the bridge to Page 3.

### Split protocol

`cs-test.csv` is unlabeled (101,503 rows) and the leaderboard closed in 2011,
so it cannot score us. State the stratified split and seed on this page.

---

## Page 3 — Modeling techniques

§3(a)(iii) requires two methods *and* why each is a reasonable candidate.
Choose a contrasting pair, not two flavors of one thing.

**Regularized logistic regression.** Binary target; outputs a calibrated
probability thresholdable against a cost ratio; coefficients are directly
interpretable. Domain justification: US adverse-action notices require reason
codes, so interpretability is a deployment constraint, not a preference.

**Gradient-boosted trees.** Captures interactions that would otherwise need
hand specification (utilization likely matters more at low income), and is
insensitive to the outliers and monotone transforms that LR requires handled —
a clean methodological contrast to the pre-processing burden documented on
Page 2.

Also settle and justify:

- **Imbalance handling** — class weights vs resampling vs threshold-moving.
  Note resampling distorts calibration, which matters if usable probabilities
  are wanted.
- **Validation protocol** — stratified k-fold, with one held-out set touched
  exactly once.

Link each method to its script or notebook (§3(a)(vi)).

---

## Page 4 — Results

**Put the majority-class baseline in the results table as a row:** 93.32%
accuracy, AUC 0.50. Most efficient demonstration of understanding in the report.

**Metrics.** ROC-AUC (the competition's own metric), PR-AUC given the 6.68%
prevalence, calibration/Brier, and a confusion matrix at the chosen threshold.

**Visualizations.** ROC overlay for both models, PR curves, calibration plot,
and — most often skipped — LR coefficients beside GBM gain importance.
Agreement between two mechanisms is a finding; disagreement is a more
interesting one.

**"Why one methodology is preferable."** Do not resolve on AUC alone. Make it
conditional on deployment: ranking applicants for human review favors the
stronger ranker; automated decisions requiring reason codes may favor LR
despite lower AUC. Quantify the gap, then say whether it justifies the
complexity.

**Sanity check (for us, not the report).** A tuned GBM lands around 0.86–0.87
AUC here; LR around 0.80–0.84 depending on transforms. A result near 0.99
means something leaked.

---

## Page 5 — Discussion

**What went well** — with magnitudes. "The sentinel indicator added *N* points
of AUC" beats "our model performed well."

**What could improve, and how** (the rubric asks for the *how*):

- No temporal dimension — only a two-year binary outcome, no deterioration
  trajectory.
- No loan-level data (amount, product type), so no exposure modeling.
- `DebtRatio` is a ratio whose denominator is missing for ~20% of rows —
  investigate whether those values mean something different.
- **No fairness audit.** `age` is a feature, and under ECOA age is a protected
  characteristic in US credit decisions.

**Future research questions:**

- Cost-sensitive thresholds using actual loss-given-default.
- A survival framing (time-to-default rather than binary-within-two-years).
- Whether the 269-row sentinel subgroup warrants its own model.

---

## Repo conventions

20% of the wiki grade is "ease of locating pertinent information," so the repo
mirrors the page order:

```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_models.ipynb
notebooks/04_results.ipynb
reports/figures/          # every figure linked from its matching wiki page
docs/wiki-outline.md      # this file
```

## Getting the data

The CSVs are not in the repo. Each teammate needs a Kaggle account, must
accept the competition rules, then:

```bash
pip install kaggle          # or: uv tool install kaggle
kaggle auth login
kaggle competitions download -c GiveMeSomeCredit -p data/raw
unzip data/raw/GiveMeSomeCredit.zip -d data/raw
```

Accepting the rules at
<https://www.kaggle.com/competitions/GiveMeSomeCredit/rules> is required —
downloads return HTTP 403 `RulesAcceptanceRequired` until you do.
