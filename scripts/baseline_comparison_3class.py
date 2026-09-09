"""
3-Class Simple Baseline Comparison (CLAUDE.md §9, §28)
Trains XGBoost and Logistic Regression baselines on the same 3-class split
to justify the complexity of TF-FaultNet.
"""
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
)
import xgboost as xgb

from src.audio_features import extract_signal_statistical_features
from src.split_manifest import load_split_manifest, rows_for_split


def main():
    print("=" * 80)
    print("   3-CLASS BASELINE COMPARISON (§9: Simple Baseline Before Complex Model)")
    print("=" * 80)

    # Load the same quarantined test set and the splits manifest
    manifest_rows = load_split_manifest("data/splits.csv")
    train_files, train_labels, _ = rows_for_split(manifest_rows, "train")
    val_files, val_labels, _ = rows_for_split(manifest_rows, "val")
    test_files, test_labels, _ = rows_for_split(manifest_rows, "test")
    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # Extract handcrafted statistical features (cached if available)
    feature_cache = "data/baseline_statistical_features_3class.npz"
    if os.path.exists(feature_cache):
        print(f"\nLoading cached baseline features from {feature_cache}...", flush=True)
        cached = np.load(feature_cache)
        X_train, X_val, X_test = cached["X_train"], cached["X_val"], cached["X_test"]
    else:
        print("\nExtracting 24 statistical audio features for baseline models...", flush=True)
        from src.dataset import load_wav
        
        def load_and_extract(filepath):
            audio, sr = load_wav(filepath)
            if len(audio) < 44100:
                audio = np.pad(audio, (0, 44100 - len(audio)))
            else:
                audio = audio[:44100]
            return extract_signal_statistical_features(audio, sr=44100)
        
        print("Extracting training features (1,374 samples)...", flush=True)
        X_train = np.array([load_and_extract(f) for f in train_files])
        print("Extracting validation features (344 samples)...", flush=True)
        X_val   = np.array([load_and_extract(f) for f in val_files])
        print("Extracting test features (430 samples)...", flush=True)
        X_test  = np.array([load_and_extract(f) for f in test_files])
        
        os.makedirs("data", exist_ok=True)
        np.savez_compressed(feature_cache, X_train=X_train, X_val=X_val, X_test=X_test)
        print(f"Cached baseline features to {feature_cache}", flush=True)

    print(f"Feature matrix shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}", flush=True)

    results = []

    # ──────────────────────────────────────────────────
    # Baseline 1: Logistic Regression (Naive Baseline)
    # ──────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("BASELINE 1: Logistic Regression (Naive Baseline)")
    print("-" * 60)

    lr_model = LogisticRegression(
        max_iter=2000, solver='lbfgs',
        C=1.0, random_state=42
    )
    lr_model.fit(X_train, train_labels)

    lr_val_preds = lr_model.predict(X_val)
    lr_test_preds = lr_model.predict(X_test)

    lr_val_acc = accuracy_score(val_labels, lr_val_preds)
    lr_val_f1 = f1_score(val_labels, lr_val_preds, average='macro')
    lr_test_acc = accuracy_score(test_labels, lr_test_preds)
    lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(test_labels, lr_test_preds, average='macro')

    print(f"  Val Accuracy:  {lr_val_acc * 100:.2f}% | Val Macro F1:  {lr_val_f1 * 100:.2f}%")
    print(f"  Test Accuracy: {lr_test_acc * 100:.2f}% | Test Macro F1: {lr_f1 * 100:.2f}%")
    print(f"  Test Precision: {lr_p * 100:.2f}% | Test Recall: {lr_r * 100:.2f}%")
    
    results.append({
        "model": "Logistic Regression",
        "type": "Naive Baseline",
        "features": "24 statistical descriptors",
        "val_accuracy": float(lr_val_acc),
        "val_macro_f1": float(lr_val_f1),
        "test_accuracy": float(lr_test_acc),
        "test_macro_precision": float(lr_p),
        "test_macro_recall": float(lr_r),
        "test_macro_f1": float(lr_f1)
    })

    # ──────────────────────────────────────────────────
    # Baseline 2: Random Forest
    # ──────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("BASELINE 2: Random Forest (Simple ML Baseline)")
    print("-" * 60)

    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, train_labels)

    rf_val_preds = rf_model.predict(X_val)
    rf_test_preds = rf_model.predict(X_test)

    rf_val_acc = accuracy_score(val_labels, rf_val_preds)
    rf_val_f1 = f1_score(val_labels, rf_val_preds, average='macro')
    rf_test_acc = accuracy_score(test_labels, rf_test_preds)
    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(test_labels, rf_test_preds, average='macro')

    print(f"  Val Accuracy:  {rf_val_acc * 100:.2f}% | Val Macro F1:  {rf_val_f1 * 100:.2f}%")
    print(f"  Test Accuracy: {rf_test_acc * 100:.2f}% | Test Macro F1: {rf_f1 * 100:.2f}%")
    print(f"  Test Precision: {rf_p * 100:.2f}% | Test Recall: {rf_r * 100:.2f}%")

    results.append({
        "model": "Random Forest (200 trees)",
        "type": "Simple ML Baseline",
        "features": "24 statistical descriptors",
        "val_accuracy": float(rf_val_acc),
        "val_macro_f1": float(rf_val_f1),
        "test_accuracy": float(rf_test_acc),
        "test_macro_precision": float(rf_p),
        "test_macro_recall": float(rf_r),
        "test_macro_f1": float(rf_f1)
    })

    # ──────────────────────────────────────────────────
    # Baseline 3: XGBoost
    # ──────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("BASELINE 3: XGBoost (Established ML Baseline)")
    print("-" * 60)

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='multi:softmax', num_class=3,
        eval_metric='mlogloss', random_state=42,
        use_label_encoder=False
    )
    xgb_model.fit(X_train, train_labels, eval_set=[(X_val, val_labels)], verbose=False)

    xgb_val_preds = xgb_model.predict(X_val)
    xgb_test_preds = xgb_model.predict(X_test)

    xgb_val_acc = accuracy_score(val_labels, xgb_val_preds)
    xgb_val_f1 = f1_score(val_labels, xgb_val_preds, average='macro')
    xgb_test_acc = accuracy_score(test_labels, xgb_test_preds)
    xgb_p, xgb_r, xgb_f1, _ = precision_recall_fscore_support(test_labels, xgb_test_preds, average='macro')

    print(f"  Val Accuracy:  {xgb_val_acc * 100:.2f}% | Val Macro F1:  {xgb_val_f1 * 100:.2f}%")
    print(f"  Test Accuracy: {xgb_test_acc * 100:.2f}% | Test Macro F1: {xgb_f1 * 100:.2f}%")
    print(f"  Test Precision: {xgb_p * 100:.2f}% | Test Recall: {xgb_r * 100:.2f}%")

    results.append({
        "model": "XGBoost (200 trees, depth=6)",
        "type": "Established ML Baseline",
        "features": "24 statistical descriptors",
        "val_accuracy": float(xgb_val_acc),
        "val_macro_f1": float(xgb_val_f1),
        "test_accuracy": float(xgb_test_acc),
        "test_macro_precision": float(xgb_p),
        "test_macro_recall": float(xgb_r),
        "test_macro_f1": float(xgb_f1)
    })

    # ──────────────────────────────────────────────────
    # TF-FaultNet (from saved report)
    # ──────────────────────────────────────────────────
    with open("results/heldout_test_evaluation_report_3class.json") as f:
        tf_report = json.load(f)

    results.append({
        "model": "TF-FaultNet (Proposed)",
        "type": "Proposed Deep Learning",
        "features": "End-to-end (raw audio → log-mel → ResNet+SE)",
        "val_accuracy": 0.9884,
        "val_macro_f1": 0.9915,
        "test_accuracy": tf_report["metrics"]["test_accuracy"],
        "test_macro_precision": tf_report["metrics"]["macro_precision"],
        "test_macro_recall": tf_report["metrics"]["macro_recall"],
        "test_macro_f1": tf_report["metrics"]["macro_f1_score"]
    })

    # ──────────────────────────────────────────────────
    # COMPARISON TABLE
    # ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("3-CLASS BASELINE COMPARISON TABLE (Same Split, Same Test Set, N=430)")
    print("=" * 100)
    print(f"{'Model':<35} | {'Type':<25} | {'Test Acc':<10} | {'Test F1':<10} | {'Test Prec':<10} | {'Test Rec':<10}")
    print("-" * 105)
    
    for r in results:
        print(f"{r['model']:<35} | {r['type']:<25} | {r['test_accuracy']*100:>8.2f}% | {r['test_macro_f1']*100:>8.2f}% | {r['test_macro_precision']*100:>8.2f}% | {r['test_macro_recall']*100:>8.2f}%")
    
    print("=" * 105)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/baseline_comparison_3class.csv", index=False)
    
    with open("results/baseline_comparison_3class.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n>>> Saved results/baseline_comparison_3class.csv")
    print(f">>> Saved results/baseline_comparison_3class.json")

    # Justify complexity
    best_baseline_f1 = max(r["test_macro_f1"] for r in results if r["type"] != "Proposed Deep Learning")
    tf_f1 = [r for r in results if r["type"] == "Proposed Deep Learning"][0]["test_macro_f1"]
    delta = (tf_f1 - best_baseline_f1) * 100
    
    print(f"\n{'='*80}")
    if delta > 0:
        print(f"§9 COMPLEXITY JUSTIFICATION: TF-FaultNet improves over best baseline by +{delta:.2f}% Macro F1")
        print(f"Deep learning complexity IS justified by observed limitations of simpler models.")
    else:
        print(f"§9 WARNING: TF-FaultNet does NOT improve over best baseline ({delta:+.2f}% Macro F1)")
        print(f"Deep learning complexity may NOT be justified.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
