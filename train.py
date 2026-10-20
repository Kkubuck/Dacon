"""LightGBM + ExtraTrees for DACON's 2022 psychological tendency competition."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "nerdiness"
DROP_COLUMNS = ["introelapse", "testelapse", "surveyelapse", "index"]
ZERO_IMPUTED = [
    "gender", "married", "education", "voted", "urban",
    "orientation", "religion", "engnat", "ASD",
]

# Best parameters recorded in the original notebook's saved Trial 4 output.
LGBM_PARAMS = {
    "num_leaves": 492,
    "max_depth": 17,
    "learning_rate": 0.01047716753520929,
    "n_estimators": 2230,
    "class_weight": None,
    "min_child_samples": 11,
    "subsample": 0.9346583262076185,
    "colsample_bytree": 0.7109969069989294,
    "reg_alpha": 0.2671838475497442,
    "reg_lambda": 0.2052238453808367,
}


def preprocess(training, *others):
    """Fit preprocessing on training only and apply it to every other frame."""
    frames = [
        frame.drop(columns=DROP_COLUMNS).copy().fillna(0)
        for frame in (training, *others)
    ]
    numeric_columns = frames[0].columns.drop("country")
    for frame in frames:
        frame[numeric_columns] = frame[numeric_columns].astype(float)
        frame.loc[frame["age"] > 80, "age"] = 0

    # As in the notebook, these means include zero-coded missing entries.
    # Reuse each fitted mean consistently for training, validation, and test.
    for column in [*ZERO_IMPUTED, "age"]:
        value = frames[0][column].mean()
        for frame in frames:
            frame.loc[frame[column] == 0, column] = value

    family_mode = frames[0]["familysize"].mode().iloc[0]
    countries = frames[0].loc[frames[0]["country"] != 0, "country"]
    common_countries = countries.value_counts().head(6).index
    country_mode = countries.mode().iloc[0] if len(countries) else "Unknown"
    for frame in frames:
        frame.loc[frame["familysize"] > 6, "familysize"] = family_mode
        frame["country"] = frame["country"].where(
            frame["country"].isin(common_countries), country_mode
        )

    encoded = [pd.get_dummies(frame, columns=["country"], dtype=float) for frame in frames]
    columns = encoded[0].columns
    return tuple(
        frame.reindex(columns=columns, fill_value=0).astype(float)
        for frame in encoded
    )


def predict(train, test, *, folds=10, seed=2022, jobs=-1,
            lgbm_estimators=2230, extra_trees_estimators=3000):
    """Return blended test probabilities; report LightGBM OOF AUC separately."""
    y = train[TARGET].astype(int)
    x = train.drop(columns=TARGET)
    if set(train[TARGET].unique()) != {0, 1}:
        raise ValueError("nerdiness must contain both binary labels 0 and 1.")
    if folds < 2 or y.value_counts().min() < folds:
        raise ValueError("Each class needs at least --folds rows, with --folds >= 2.")
    if min(lgbm_estimators, extra_trees_estimators) < 1:
        raise ValueError("Estimator counts must be positive.")

    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    fold_predictions = []
    for fold, (fit_index, valid_index) in enumerate(split.split(x, y), start=1):
        x_fit, x_valid, x_test = preprocess(
            x.iloc[fit_index], x.iloc[valid_index], test
        )
        params = {**LGBM_PARAMS, "n_estimators": lgbm_estimators}
        model = LGBMClassifier(
            **params, objective="binary", random_state=seed, n_jobs=jobs, verbosity=-1
        )
        model.fit(x_fit, y.iloc[fit_index])
        valid_prediction = model.predict_proba(x_valid)[:, 1]
        oof[valid_index] = valid_prediction
        fold_predictions.append(model.predict_proba(x_test)[:, 1])
        score = roc_auc_score(y.iloc[valid_index], valid_prediction)
        print(f"LightGBM fold {fold}/{folds}: ROC-AUC {score:.5f}")

    lgbm_prediction = np.mean(fold_predictions, axis=0)
    print(f"LightGBM OOF ROC-AUC: {roc_auc_score(y, oof):.5f}")

    x_full, x_test = preprocess(x, test)
    extra_trees = ExtraTreesClassifier(
        n_estimators=extra_trees_estimators, random_state=seed, n_jobs=jobs
    )
    extra_trees.fit(x_full, y)
    extra_prediction = extra_trees.predict_proba(x_test)[:, 1]
    return 0.6 * extra_prediction + 0.4 * lgbm_prediction


def load_data(data_dir):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    submission = pd.read_csv(data_dir / "sample_submission.csv")
    required = {*DROP_COLUMNS, *ZERO_IMPUTED, "age", "familysize", "country"}
    for name, frame in [("train", train), ("test", test)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
        if frame.empty or frame["index"].isna().any() or not frame["index"].is_unique:
            raise ValueError(f"{name}.csv must have nonempty, unique, non-null IDs.")
    if TARGET not in train or train[TARGET].isna().any():
        raise ValueError("train.csv must contain non-null nerdiness labels.")
    if set(train.columns) - {TARGET} != set(test.columns):
        raise ValueError("Train and test feature columns must match.")
    if set(submission.columns) != {"index", TARGET}:
        raise ValueError("sample_submission.csv must contain index and nerdiness.")
    if (not submission["index"].is_unique
            or set(submission["index"]) != set(test["index"])):
        raise ValueError("Sample submission IDs must match test IDs exactly.")
    return train, test, submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--lgbm-estimators", type=int, default=2230)
    parser.add_argument("--extra-trees-estimators", type=int, default=3000)
    args = parser.parse_args()

    train, test, submission = load_data(args.data_dir)
    prediction = predict(
        train, test, folds=args.folds, seed=args.seed, jobs=args.jobs,
        lgbm_estimators=args.lgbm_estimators,
        extra_trees_estimators=args.extra_trees_estimators,
    )
    if not np.isfinite(prediction).all() or not ((prediction >= 0) & (prediction <= 1)).all():
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    submission[TARGET] = pd.Series(
        prediction, index=test["index"]
    ).reindex(submission["index"]).to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
