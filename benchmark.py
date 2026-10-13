#!/usr/bin/env python3
"""
Benchmark script for health app churn dataset.
Runs 5-fold stratified cross-validation for multiple models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
import sys
import os
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """
    Load dataset and separate features and target.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Separate features and target
    X = df.drop(['churned', 'user_id'], axis=1)  # Drop ID and target
    y = df['churned']

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

    return X, y

def create_preprocessing_pipeline(X):
    """
    Create preprocessing pipeline for mixed data types.
    """
    # Identify column types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Numeric pipeline: impute median, scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: impute most frequent, one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def evaluate_model(model, X, y, cv_folds=5):
    """
    Evaluate model using stratified cross-validation.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create and fit pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', create_preprocessing_pipeline(X_train)),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)

        # Predict probabilities
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

        # Calculate AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        auc_scores.append(auc)

        print(f"  Fold {fold+1}: AUC = {auc:.4f}")

    return np.mean(auc_scores), np.std(auc_scores)

def run_benchmarks(filepath):
    """
    Run benchmarks for all specified models.
    """
    # Load data
    X, y = load_and_preprocess_data(filepath)

    # Define models to evaluate
    models = {
        'Majority Class Baseline': None,  # Special case
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gaussian Naive Bayes': GaussianNB(),
        'Decision Tree (max_depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingClassifier(random_state=42)
    }

    results = {}

    print("\nRunning 5-fold stratified cross-validation...")
    print("=" * 50)

    for name, model in models.items():
        print(f"\n{name}:")

        if name == 'Majority Class Baseline':
            # For majority class, AUC is always 0.5 for binary classification
            # since it doesn't rank predictions - it always predicts the same class
            mean_auc = 0.5
            std_auc = 0.0
            print(f"  AUC = {mean_auc:.4f} (+/- {std_auc:.4f})")
        else:
            mean_auc, std_auc = evaluate_model(model, X, y)
            print(f"  AUC = {mean_auc:.4f} (+/- {std_auc:.4f})")

        results[name] = {'mean_auc': mean_auc, 'std_auc': std_auc}

    return results

def print_results_table(results):
    """
    Print results in a formatted table.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS - 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)
    print(f"{'Model':<30} {'Mean AUC':<12} {'Std AUC':<12}")
    print("-" * 60)

    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['mean_auc']:<12.4f} {metrics['std_auc']:<12.4f}")

    print("-" * 60)

    # Check if targets are met
    print("\nTARGET VERIFICATION:")
    print("-" * 30)

    lr_auc = results['Logistic Regression']['mean_auc']
    rf_auc = results['Random Forest']['mean_auc']
    gb_auc = results['Gradient Boosting']['mean_auc']
    nb_auc = results['Gaussian Naive Bayes']['mean_auc']
    dt_auc = results['Decision Tree (max_depth=5)']['mean_auc']
    baseline_auc = results['Majority Class Baseline']['mean_auc']

    # Target 1: Best model ROC-AUC between 0.85 and 0.93
    best_auc = max(rf_auc, gb_auc)
    target1_met = 0.85 <= best_auc <= 0.93
    print(f"1. Best model AUC in [0.85, 0.93]: {best_auc:.4f} {'PASS' if target1_met else 'FAIL'}")

    # Target 2: Nothing near 1.0 or 0.5
    not_near_1 = best_auc < 0.95
    not_near_05 = lr_auc > 0.55  # Should be clearly better than baseline
    target2_met = not_near_1 and not_near_05
    print(f"2. Not near 1.0 or 0.5: Best={best_auc:.4f}<0.95, LR={lr_auc:.4f}>0.55 {'PASS' if target2_met else 'FAIL'}")

    # Target 3: GB and RF beat LR by at least 0.02
    target3_rf = (rf_auc - lr_auc) >= 0.02
    target3_gb = (gb_auc - lr_auc) >= 0.02
    target3_met = target3_rf and target3_gb
    print(f"3. RF and GB >= LR + 0.02: RF diff={rf_auc-lr_auc:.4f}, GB diff={gb_auc-lr_auc:.4f} {'PASS' if target3_met else 'FAIL'}")

    # Target 4: NB and DT-5 clearly below best model
    target4_nb = (best_auc - nb_auc) >= 0.10  # Clearly below
    target4_dt = (best_auc - dt_auc) >= 0.05  # Reasonably below
    target4_met = target4_nb and target4_dt
    print(f"4. NB and DT clearly below best: NB diff={best_auc-nb_auc:.4f}, DT diff={best_auc-dt_auc:.4f} {'PASS' if target4_met else 'FAIL'}")

    # Target 5: Gaps between top models > CV std dev
    top_models_sorted = sorted([('RF', rf_auc, results['Random Forest']['std_auc']),
                                ('GB', gb_auc, results['Gradient Boosting']['std_auc'])],
                               key=lambda x: x[1], reverse=True)
    if len(top_models_sorted) >= 2:
        gap = top_models_sorted[0][1] - top_models_sorted[1][1]
        avg_std = (top_models_sorted[0][2] + top_models_sorted[1][2]) / 2
        target5_met = gap > avg_std
        print(f"5. Top model gap > avg std: Gap={gap:.4f}, Avg std={avg_std:.4f} {'PASS' if target5_met else 'FAIL'}")
    else:
        target5_met = False
        print("5. Cannot compute gap (need at least 2 models) FAIL")

    all_targets_met = target1_met and target2_met and target3_met and target4_met and target5_met
    print(f"\nALL TARGETS MET: {'YES' if all_targets_met else 'NO'}")

    return all_targets_met, results

def main():
    """
    Main function to run benchmarks.
    """
    # Default dataset path
    dataset_path = "health_app_churn_50k.csv"

    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset file '{dataset_path}' not found.")
        print("Please run generate_dataset.py first to create the dataset.")
        sys.exit(1)

    # Run benchmarks
    results = run_benchmarks(dataset_path)

    # Print results table and check targets
    targets_met, results_dict = print_results_table(results)

    # Save results to file
    output_file = "benchmark_results.txt"
    print(f"\nSaving detailed results to {output_file}...")

    with open(output_file, 'w') as f:
        f.write("Health App Churn Dataset Benchmark Results\n")
        f.write("=" * 50 + "\n\n")

        f.write("Dataset: health_app_churn_50k.csv\n")
        f.write("Models evaluated: Majority Class Baseline, Logistic Regression, Gaussian Naive Bayes, ")
        f.write("Decision Tree (max_depth=5), Random Forest, Gradient Boosting (HistGradientBoosting)\n")
        f.write("Validation: 5-fold stratified cross-validation\n")
        f.write("Metric: ROC-AUC\n\n")

        f.write("RESULTS TABLE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Model':<30} {'Mean AUC':<12} {'Std AUC':<12}\n")
        f.write("-" * 30 + "\n")
        for model_name, metrics in results_dict.items():
            f.write(f"{model_name:<30} {metrics['mean_auc']:<12.4f} {metrics['std_auc']:<12.4f}\n")
        f.write("-" * 30 + "\n\n")

        f.write("TARGET VERIFICATION:\n")
        f.write("-" * 20 + "\n")
        lr_auc = results_dict['Logistic Regression']['mean_auc']
        rf_auc = results_dict['Random Forest']['mean_auc']
        gb_auc = results_dict['Gradient Boosting']['mean_auc']
        nb_auc = results_dict['Gaussian Naive Bayes']['mean_auc']
        dt_auc = results_dict['Decision Tree (max_depth=5)']['mean_auc']
        baseline_auc = results_dict['Majority Class Baseline']['mean_auc']

        best_auc = max(rf_auc, gb_auc)
        f.write(f"1. Best model AUC in [0.85, 0.93]: {best_auc:.4f}\n")
        f.write(f"2. Not near 1.0 or 0.5: Best={best_auc:.4f}, LR={lr_auc:.4f}\n")
        f.write(f"3. RF and GB >= LR + 0.02: RF diff={rf_auc-lr_auc:.4f}, GB diff={gb_auc-lr_auc:.4f}\n")
        f.write(f"4. NB and DT clearly below best: NB diff={best_auc-nb_auc:.4f}, DT diff={best_auc-dt_auc:.4f}\n")

        top_models_sorted = sorted([('RF', rf_auc, results_dict['Random Forest']['std_auc']),
                                    ('GB', gb_auc, results_dict['Gradient Boosting']['std_auc'])],
                                   key=lambda x: x[1], reverse=True)
        if len(top_models_sorted) >= 2:
            gap = top_models_sorted[0][1] - top_models_sorted[1][1]
            avg_std = (top_models_sorted[0][2] + top_models_sorted[1][2]) / 2
            f.write(f"5. Top model gap > avg std: Gap={gap:.4f}, Avg std={avg_std:.4f}\n")

        f.write(f"\nOverall: ALL TARGETS MET: {'YES' if targets_met else 'NO'}\n")

    print(f"Results saved to {output_file}")

    # Print final summary
    print("\n" + "=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"Dataset: {dataset_path}")
    print(f"Results: {output_file}")
    if targets_met:
        print("PASS: All benchmark targets were met!")
    else:
        print("FAIL: Some benchmark targets were not met.")
        print("   Consider adjusting the data generation parameters.")

if __name__ == "__main__":
    main()