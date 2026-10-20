# DACON 2022 · Psychological Tendency Prediction

**Private 1st place · ROC-AUC 0.90396 · Team AIM. Lab.**

LightGBM and ExtraTrees solution for the **SW-Centered University Joint AI Competition — Preliminary Round**.

[Competition](https://dacon.io/competitions/official/235902/overview/description) · [Original code share](https://dacon.io/competitions/official/235902/codeshare/6153) · [Original notebook](notebooks/original_submission.ipynb)

## Method

- Replace missing values and selected outliers using training statistics; one-hot encode the six most frequent countries.
- Train LightGBM on 10 stratified folds using the parameters recorded in the original notebook, and average all fold predictions.
- Train ExtraTrees with 3,000 trees on the full training set.
- Blend probabilities: **0.6 × ExtraTrees + 0.4 × LightGBM**.

The blend follows the submitted code and the author's correction in the DACON comments.

## Run

Tested with Python 3.12. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the competition data from [DACON](https://dacon.io/competitions/official/235902/data) and place these files in `data/`:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

```bash
python train.py --data-dir data --output outputs/submission.csv
```

The script prints LightGBM out-of-fold ROC-AUC and writes probabilities for `nerdiness` in the sample submission's ID order. Data and generated submissions are excluded from Git.

## Repository

- `train.py`: standalone training and submission script; no Colab or Optuna search required.
- `notebooks/original_submission.ipynb`: historical notebook, with outputs and Colab metadata removed. Its original code is preserved.
- `requirements.txt`: pinned dependencies for the cleaned script, not the original 2022 Colab environment.

The runner fixes the original notebook's reused fold-training variables and skipped fold during prediction averaging. Preprocessing is fitted separately on each training fold, and train/test dummy columns are aligned. **0.90396 is the historical private score**; the cleaned runner has not been re-scored on DACON and can produce different predictions.

## Acknowledgments

References credited in the original notebook:

- 유재성 KADE — [LightGBM + Optuna로 top 10안에 들어봅시다.](https://dacon.io/codeshare/2876)
- SDSTony — [정형데이터는 AutoML에 때려박고(?) 시작하자!](https://dacon.io/codeshare/1701)
