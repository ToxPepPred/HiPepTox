import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import extractfeature


FEATURES = [
    "AAC",
    "DPC",
    "NBP",
    "CBP",
    "pKa1",
    "pKa2",
    "Vol",
    "VSC",
    "SASA",
    "pI",
    "Chg",
    "NCIS",
    "Hyd1",
    "Hyd2",
    "Hydro",
    "FEtmh",
    "Pol2",
]

STANDARDIZE_FEATURES = [
    "pKa1",
    "pKa2",
    "Vol",
    "VSC",
    "SA",
    "pI",
    "Chg",
    "NCIS",
    "Hyd1",
    "Hyd2",
    "Hydro",
    "FEtmh",
    "Pol2",
]

AA_SET = set(extractfeature.aaLi)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ToxTeller-style models from labeled CSV and evaluate on an external labeled CSV."
    )
    parser.add_argument("--train_csv", required=True, help="Path to the training CSV.")
    parser.add_argument("--test_csv", required=True, help="Path to the external test CSV.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save trained models, predictions, and metrics. Defaults to a timestamped directory.",
    )
    parser.add_argument("--min_len", type=int, default=10, help="Minimum peptide length supported by ToxTeller.")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum peptide length supported by ToxTeller.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def ensure_output_dir(output_dir, train_csv):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(os.path.dirname(os.path.abspath(train_csv))) or "dataset"
    auto_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results",
        f"{dataset_name}_train_eval_{timestamp}",
    )
    os.makedirs(auto_dir, exist_ok=True)
    return auto_dir


def coerce_label(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "positive", "pos", "toxic", "tox"}:
        return 1
    if text in {"0", "false", "negative", "neg", "non-toxic", "nontoxic", "non_toxic"}:
        return 0
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Unsupported toxicity label: {value}") from exc


def valid_sequence(sequence):
    return bool(sequence) and set(sequence).issubset(AA_SET)


def load_and_filter_dataset(csv_path, min_len, max_len):
    df = pd.read_csv(csv_path)
    if "sequence" not in df.columns or "toxicity" not in df.columns:
        raise ValueError(f"{csv_path} must contain `sequence` and `toxicity` columns.")

    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["label"] = df["toxicity"].map(coerce_label)
    df["length"] = df["sequence"].str.len()
    df["valid_chars"] = df["sequence"].map(valid_sequence)
    df["valid_length"] = df["length"].between(min_len, max_len)

    kept = df[df["valid_chars"] & df["valid_length"]].copy().reset_index(drop=True)
    kept["row_index"] = kept.index
    if "id" not in kept.columns:
        kept["id"] = [f"sample_{idx:06d}" for idx in range(len(kept))]
    kept["sample_name"] = [
        f"{'pos' if label == 1 else 'neg'}_{idx:06d}_{sample_id}"
        for idx, (label, sample_id) in enumerate(zip(kept["label"], kept["id"]))
    ]

    summary = {
        "csv_path": csv_path,
        "total_rows": int(len(df)),
        "kept_rows": int(len(kept)),
        "filtered_invalid_chars": int((~df["valid_chars"]).sum()),
        "filtered_out_of_length_range": int((~df["valid_length"]).sum()),
        "positive_rows_kept": int((kept["label"] == 1).sum()),
        "negative_rows_kept": int((kept["label"] == 0).sum()),
        "min_len": int(min_len),
        "max_len": int(max_len),
    }
    return kept, summary


def build_feature_matrix(df):
    seq_dict = dict(zip(df["sample_name"], df["sequence"]))
    feature_df = extractfeature.GenerateFeatureTableGivenSeqDiAndFeatureLi(
        seq_dict,
        FEATURES,
        islabeled=True,
    )
    meta_df = df[["sample_name", "id", "sequence", "toxicity", "label", "length"]].copy()
    feature_df = feature_df.merge(
        meta_df,
        left_on="Name",
        right_on="sample_name",
        how="left",
        validate="one_to_one",
    )

    excluded_columns = {"Name", "Sequence", "Length", "TrueLabel", "sample_name", "id", "sequence", "toxicity", "label", "length"}
    x_data = feature_df[[col for col in feature_df.columns if col not in excluded_columns]].copy()
    y_data = feature_df["label"].astype(int).to_numpy()
    return feature_df, x_data, y_data


def apply_scaler(train_x, test_x):
    scale_cols = [col for col in STANDARDIZE_FEATURES if col in train_x.columns]
    scaler = StandardScaler()

    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_x[scale_cols]),
        columns=scale_cols,
        index=train_x.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_x[scale_cols]),
        columns=scale_cols,
        index=test_x.index,
    )

    train_rest = train_x.drop(columns=scale_cols)
    test_rest = test_x.drop(columns=scale_cols)

    train_final = pd.concat([train_scaled, train_rest], axis=1)
    test_final = pd.concat([test_scaled, test_rest], axis=1)
    return scaler, train_final, test_final


def build_models(random_state):
    return {
        "lr": LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            random_state=random_state,
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            cache_size=1024,
            random_state=random_state,
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_pickle(obj, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir, args.train_csv)

    train_df, train_summary = load_and_filter_dataset(args.train_csv, args.min_len, args.max_len)
    test_df, test_summary = load_and_filter_dataset(args.test_csv, args.min_len, args.max_len)

    train_feature_df, train_x_raw, y_train = build_feature_matrix(train_df)
    test_feature_df, test_x_raw, y_test = build_feature_matrix(test_df)

    feature_columns = list(train_x_raw.columns)
    test_x_raw = test_x_raw.reindex(columns=feature_columns)

    metrics_by_model = {}
    prediction_table = test_df[["id", "sequence", "toxicity", "label", "length"]].copy()
    trained_artifacts = {}

    for model_name, model in build_models(args.random_state).items():
        scaler, train_x, test_x = apply_scaler(train_x_raw, test_x_raw)
        model.fit(train_x, y_train)

        model_path = os.path.join(output_dir, f"model_{model_name}.pkl")
        scaler_path = os.path.join(output_dir, f"scaler_{model_name}.pkl")
        save_pickle(model, model_path)
        save_pickle(scaler, scaler_path)

        loaded_model = load_pickle(model_path)
        loaded_scaler = load_pickle(scaler_path)
        reloaded_train_x, reloaded_test_x = None, None

        scale_cols = [col for col in STANDARDIZE_FEATURES if col in train_x_raw.columns]
        train_scaled = pd.DataFrame(
            loaded_scaler.transform(train_x_raw[scale_cols]),
            columns=scale_cols,
            index=train_x_raw.index,
        )
        test_scaled = pd.DataFrame(
            loaded_scaler.transform(test_x_raw[scale_cols]),
            columns=scale_cols,
            index=test_x_raw.index,
        )
        train_rest = train_x_raw.drop(columns=scale_cols)
        test_rest = test_x_raw.drop(columns=scale_cols)
        reloaded_train_x = pd.concat([train_scaled, train_rest], axis=1)
        reloaded_test_x = pd.concat([test_scaled, test_rest], axis=1)

        train_pred = loaded_model.predict(reloaded_train_x)
        test_pred = loaded_model.predict(reloaded_test_x)
        metrics_by_model[model_name] = {
            "train": compute_metrics(y_train, train_pred),
            "external_test": compute_metrics(y_test, test_pred),
            "model_path": model_path,
            "scaler_path": scaler_path,
        }
        prediction_table[f"{model_name}_pred"] = test_pred.astype(int)
        trained_artifacts[model_name] = {
            "model_path": model_path,
            "scaler_path": scaler_path,
        }

    best_model_name = max(
        metrics_by_model,
        key=lambda name: metrics_by_model[name]["external_test"]["mcc"],
    )

    with open(os.path.join(output_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": feature_columns,
                "standardized_columns": [col for col in STANDARDIZE_FEATURES if col in train_x_raw.columns],
                "considered_features": FEATURES,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    prediction_csv_path = os.path.join(output_dir, "external_test_predictions.csv")
    prediction_table.to_csv(prediction_csv_path, index=False)

    summary = {
        "train_dataset": train_summary,
        "external_test_dataset": test_summary,
        "models": metrics_by_model,
        "best_model_by_external_mcc": best_model_name,
        "best_model_external_metrics": metrics_by_model[best_model_name]["external_test"],
        "artifacts": trained_artifacts,
        "predictions_csv": prediction_csv_path,
    }
    summary_path = os.path.join(output_dir, "train_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
