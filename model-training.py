import argparse
import contextlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier

SMOKING_MAP = {
    "No Info": 0,
    "never": 1,
    "not current": 2,
    "ever": 3,
    "former": 4,
    "current": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train diabetes model with drift-aware report")
    parser.add_argument(
        "--data",
        default="diabetes_prediction_dataset.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--output-model",
        default="best_diabetes_model.pkl",
        help="Output model artifact path",
    )
    parser.add_argument(
        "--report",
        default="reports/model_training_analysis.md",
        help="Markdown report output path",
    )
    parser.add_argument(
        "--report-html",
        default="reports/model_training_analysis.html",
        help="HTML report output path",
    )
    parser.add_argument(
        "--metrics-json",
        default="reports/model_training_metrics.json",
        help="Metrics JSON output path",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resample",
        choices=["none", "oversample"],
        default="oversample",
        help="Resampling strategy to address class imbalance",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="mlruns",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="diabetes_prediction",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    return parser.parse_args()


def oversample_minority(X, y, random_state):
    df = X.copy()
    df["__target__"] = y

    majority = df[df["__target__"] == 0]
    minority = df[df["__target__"] == 1]

    if len(minority) == 0 or len(majority) == 0:
        return X, y

    minority_up = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=random_state,
    )
    upsampled = pd.concat([majority, minority_up], axis=0)
    upsampled = upsampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_res = upsampled.pop("__target__")
    return upsampled, y_res


def make_markdown_table(rows, headers):
    if not rows:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(row.get(h, "")) for h in headers) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body_lines)


def slugify(value):
        return value.lower().replace(" ", "_").replace("-", "_")


def write_html_report(markdown_text, output_path):
        escaped = html.escape(markdown_text)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <title>Model Training Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        pre {{ white-space: pre-wrap; font-family: Menlo, Consolas, monospace; }}
        .content {{ border: 1px solid #e0e0e0; padding: 16px; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>Model Training Analysis</h1>
    <div class=\"content\">
        <pre>{escaped}</pre>
    </div>
</body>
</html>"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html_content, encoding="utf-8")


def main():
    args = parse_args()
    mlflow_enabled = not args.disable_mlflow
    if mlflow_enabled:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    report_lines = []

    def log(line=""):
        report_lines.append(line)

    def maybe_start_run(name, nested=False):
        if not mlflow_enabled:
            return contextlib.nullcontext()
        return mlflow.start_run(run_name=name, nested=nested)

    if mlflow_enabled:
        mlflow.start_run(run_name="training-pipeline")
        mlflow.log_param("data_path", args.data)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("resample", args.resample)

    df = pd.read_csv(args.data)

    log("# Model Training Analysis")
    log("")
    log("## Data Summary")
    log(f"Rows: {len(df)}")
    log(f"Columns: {len(df.columns)}")
    log("")
    log("Missing values:")
    missing = df.isnull().sum().to_dict()
    for key, value in missing.items():
        log(f"- {key}: {value}")
    log("")
    class_counts = df["diabetes"].value_counts().to_dict()
    pos_rate = df["diabetes"].mean()
    log("Class distribution:")
    log(f"- 0 (No Diabetes): {class_counts.get(0, 0)}")
    log(f"- 1 (Diabetes): {class_counts.get(1, 0)}")
    log(f"- Positive rate: {pos_rate * 100:.2f}%")

    if mlflow_enabled:
        mlflow.log_param("rows", len(df))
        mlflow.log_param("columns", len(df.columns))
        mlflow.log_param("class_0_count", class_counts.get(0, 0))
        mlflow.log_param("class_1_count", class_counts.get(1, 0))
        mlflow.log_metric("positive_rate", float(pos_rate))

    df_proc = df.copy()
    df_proc = df_proc[df_proc["gender"] != "Other"].reset_index(drop=True)

    gender_encoder = LabelEncoder()
    df_proc["gender"] = gender_encoder.fit_transform(df_proc["gender"])
    df_proc["smoking_history"] = df_proc["smoking_history"].map(SMOKING_MAP)

    if df_proc["smoking_history"].isna().any():
        missing_values = df_proc[df_proc["smoking_history"].isna()]["smoking_history"].unique()
        raise ValueError(f"Unknown smoking_history values: {missing_values}")

    log("")
    log("## Preprocessing")
    log("- Dropped gender = Other")
    log("- Encoded gender with LabelEncoder (Female=0, Male=1)")
    log("- Encoded smoking_history with ordinal mapping")

    df_proc["bmi_category"] = pd.cut(
        df_proc["bmi"],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=[0, 1, 2, 3],
    ).astype(int)

    df_proc["age_group"] = pd.cut(
        df_proc["age"],
        bins=[0, 30, 45, 60, 100],
        labels=[0, 1, 2, 3],
    ).astype(int)

    df_proc["risk_score"] = (
        df_proc["hypertension"]
        + df_proc["heart_disease"]
        + (df_proc["bmi"] > 30).astype(int)
        + (df_proc["HbA1c_level"] >= 5.7).astype(int)
        + (df_proc["blood_glucose_level"] >= 126).astype(int)
    )
    df_proc["age_bmi"] = df_proc["age"] * df_proc["bmi"]

    log("")
    log("## Feature Engineering")
    log("- Added bmi_category, age_group, risk_score, age_bmi")

    X_all = df_proc.drop("diabetes", axis=1)
    y_all = df_proc["diabetes"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.15,
        random_state=args.random_state,
        stratify=y_all,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.176,
        random_state=args.random_state,
        stratify=y_temp,
    )

    mi_scores = mutual_info_classif(X_train, y_train, random_state=args.random_state)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

    selected_features = mi_series[mi_series > 0.01].index.tolist()

    log("")
    log("## Feature Selection")
    log("- MI computed on training split only (prevents leakage)")
    log(f"Selected {len(selected_features)} features (MI > 0.01):")
    log(", ".join(selected_features))

    if mlflow_enabled:
        mlflow.log_param("selected_features_count", len(selected_features))

    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    log("")
    log("## Split Strategy")
    log(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    log(f"Train positive rate: {y_train.mean() * 100:.2f}%")
    log(f"Val positive rate: {y_val.mean() * 100:.2f}%")
    log(f"Test positive rate: {y_test.mean() * 100:.2f}%")

    if mlflow_enabled:
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_metric("train_positive_rate", float(y_train.mean()))
        mlflow.log_metric("val_positive_rate", float(y_val.mean()))
        mlflow.log_metric("test_positive_rate", float(y_test.mean()))

    if args.resample == "oversample":
        X_train_res, y_train_res = oversample_minority(X_train, y_train, args.random_state)
        log("")
        log("## Resampling Strategy")
        log("- Oversampled minority class on training set only (prevents data leakage)")
        log(f"Resampled train size: {len(X_train_res)}")
        log(f"Resampled positive rate: {y_train_res.mean() * 100:.2f}%")
        if mlflow_enabled:
            mlflow.log_param("resampled_train_size", len(X_train_res))
            mlflow.log_metric("resampled_positive_rate", float(y_train_res.mean()))
    else:
        X_train_res, y_train_res = X_train.copy(), y_train.copy()
        log("")
        log("## Resampling Strategy")
        log("- None")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_res)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    log("")
    log("## Scaling")
    log("- StandardScaler fit on training data only (prevents leakage)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scoring = ["accuracy", "roc_auc", "f1", "precision", "recall"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=args.random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=args.random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=args.random_state,
            eval_metric="logloss",
            verbosity=0,
        ),
    }

    cv_results = {}
    for name, model in models.items():
        scores = cross_validate(model, X_train_s, y_train_res, cv=cv, scoring=cv_scoring, n_jobs=-1)
        cv_results[name] = {
            "Accuracy": scores["test_accuracy"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean(),
            "F1": scores["test_f1"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
        }

    if mlflow_enabled:
        for name, metrics in cv_results.items():
            prefix = slugify(name)
            for metric_name, value in metrics.items():
                metric_key = f"cv_{prefix}_{slugify(metric_name)}"
                mlflow.log_metric(metric_key, float(value))

    log("")
    log("## 5-Fold CV Results")
    cv_rows = [
        {"Model": name, **{k: f"{v:.4f}" for k, v in metrics.items()}}
        for name, metrics in cv_results.items()
    ]
    log(make_markdown_table(cv_rows, ["Model", "Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]))

    val_results = {}
    for name, model in models.items():
        with maybe_start_run(name, nested=True):
            if mlflow_enabled:
                raw_params = model.get_params()
                if name == "Logistic Regression":
                    keys = ["C", "max_iter", "solver"]
                elif name == "Random Forest":
                    keys = ["n_estimators", "max_depth"]
                else:
                    keys = ["n_estimators", "learning_rate", "max_depth"]
                mlflow.log_params({key: raw_params.get(key) for key in keys})
                mlflow.log_param("n_features", len(selected_features))
                mlflow.log_param("train_size", len(X_train_res))

            model.fit(X_train_s, y_train_res)
            y_pred = model.predict(X_val_s)
            y_proba = model.predict_proba(X_val_s)[:, 1]
            val_metrics = {
                "Accuracy": accuracy_score(y_val, y_pred),
                "ROC-AUC": roc_auc_score(y_val, y_proba),
                "F1": f1_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred),
                "Recall": recall_score(y_val, y_pred),
            }
            val_results[name] = val_metrics

            if mlflow_enabled:
                mlflow.log_metrics({
                    "val_accuracy": val_metrics["Accuracy"],
                    "val_roc_auc": val_metrics["ROC-AUC"],
                    "val_f1": val_metrics["F1"],
                    "val_precision": val_metrics["Precision"],
                    "val_recall": val_metrics["Recall"],
                })
                if name == "XGBoost":
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")

    log("")
    log("## Validation Results")
    val_rows = [
        {"Model": name, **{k: f"{v:.4f}" for k, v in metrics.items()}}
        for name, metrics in val_results.items()
    ]
    log(make_markdown_table(val_rows, ["Model", "Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]))

    xgb_sweep_results = []
    for lr in [0.01, 0.05, 0.1, 0.3]:
        with maybe_start_run(f"XGBoost_lr_{lr}", nested=True):
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=lr,
                max_depth=6,
                random_state=args.random_state,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X_train_s, y_train_res)
            y_pred = model.predict(X_val_s)
            y_proba = model.predict_proba(X_val_s)[:, 1]
            sweep_metrics = {
                "ROC-AUC": roc_auc_score(y_val, y_proba),
                "F1": f1_score(y_val, y_pred),
            }
            xgb_sweep_results.append({
                "LR": lr,
                "ROC-AUC": f"{sweep_metrics['ROC-AUC']:.4f}",
                "F1": f"{sweep_metrics['F1']:.4f}",
            })

            if mlflow_enabled:
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 6)
                mlflow.log_metrics({
                    "val_roc_auc": sweep_metrics["ROC-AUC"],
                    "val_f1": sweep_metrics["F1"],
                })
                mlflow.xgboost.log_model(model, "model")

    log("")
    log("## XGBoost Learning-Rate Sweep")
    log(make_markdown_table(xgb_sweep_results, ["LR", "ROC-AUC", "F1"]))

    val_df = pd.DataFrame(val_results).T
    best_model_name = val_df["ROC-AUC"].idxmax()
    best_model = models[best_model_name]

    log("")
    log("## Model Selection")
    log(f"Best model by validation ROC-AUC: {best_model_name}")

    if mlflow_enabled:
        mlflow.log_param("best_model", best_model_name)

    best_model.fit(X_train_s, y_train_res)
    y_test_pred = best_model.predict(X_test_s)
    y_test_proba = best_model.predict_proba(X_test_s)[:, 1]

    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "ROC-AUC": roc_auc_score(y_test, y_test_proba),
        "F1": f1_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
    }

    with maybe_start_run("best-model", nested=True):
        if mlflow_enabled:
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_metrics({
                "test_accuracy": test_metrics["Accuracy"],
                "test_roc_auc": test_metrics["ROC-AUC"],
                "test_f1": test_metrics["F1"],
                "test_precision": test_metrics["Precision"],
                "test_recall": test_metrics["Recall"],
            })
            if best_model_name == "XGBoost":
                mlflow.xgboost.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")

    log("")
    log("## Test Set Results")
    for key, value in test_metrics.items():
        log(f"- {key}: {value:.4f}")

    report_lines.append("")
    report_lines.append("### Classification Report")
    report_lines.append("```")
    report_lines.append(classification_report(y_test, y_test_pred, target_names=["No Diabetes", "Diabetes"]))
    report_lines.append("```")

    cm = confusion_matrix(y_test, y_test_pred).tolist()

    report_lines.append("")
    report_lines.append("### Confusion Matrix")
    report_lines.append("```")
    report_lines.append(str(cm))
    report_lines.append("```")

    inference_bundle = {
        "model": best_model,
        "scaler": scaler,
        "selected_features": selected_features,
        "feature_columns": list(X_train.columns),
    }

    joblib.dump(inference_bundle, args.output_model)

    log("")
    log("## Artifact")
    log(f"Saved model bundle to: {args.output_model}")

    metrics_payload = {
        "cv_results": cv_results,
        "val_results": val_results,
        "test_results": test_metrics,
        "confusion_matrix": cm,
        "selected_features": selected_features,
        "best_model": best_model_name,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(report_lines)
    report_path.write_text(report_text, encoding="utf-8")

    html_report_path = Path(args.report_html)
    write_html_report(report_text, html_report_path)

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if mlflow_enabled:
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(html_report_path))
        mlflow.log_artifact(str(metrics_path))

    print("Model training complete.")
    print(f"Report written to {report_path}")
    print(f"HTML report written to {html_report_path}")
    print(f"Metrics JSON written to {metrics_path}")

    if mlflow_enabled:
        mlflow.end_run()


if __name__ == "__main__":
    main()
