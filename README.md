# DACON 2022 · Psychological Tendency Prediction

SW-Centered University Joint AI Competition — Preliminary Round

**Team AIM. Lab. · Hanbat National University**

[Competition](https://dacon.io/competitions/official/235902/overview/description) · [Leaderboard](https://dacon.io/competitions/official/235902/leaderboard) · [Code share](https://dacon.io/competitions/official/235902/codeshare/6153) · [Notebook](notebooks/original_submission.ipynb)

## Competition Results

| Leaderboard | Rank | ROC-AUC |
| :--- | ---: | ---: |
| Public | 2nd | 0.90770 |
| **Private** | **1st** | **0.90396** |

## Method

**0.6 × ExtraTrees + 0.4 × LightGBM**

- ExtraTrees: 3,000 trees.
- LightGBM: 10-fold stratified cross-validation.

## Run

Python 3.12. Place `train.csv`, `test.csv`, and `sample_submission.csv` from [DACON](https://dacon.io/competitions/official/235902/data) in `data/`.

```bash
python -m pip install -r requirements.txt
python train.py --data-dir data --output outputs/submission.csv
```

## References

- 유재성 KADE — [LightGBM + Optuna로 top 10안에 들어봅시다.](https://dacon.io/codeshare/2876)
- SDSTony — [정형데이터는 AutoML에 때려박고(?) 시작하자!](https://dacon.io/codeshare/1701)
