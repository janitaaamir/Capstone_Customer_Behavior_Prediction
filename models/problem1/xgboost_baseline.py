import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             precision_recall_curve, roc_curve)

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


def find_threshold(y_true, y_probs, min_precision=0.3):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    thresholds = np.append(thresholds, 1.0)

    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= min_precision and r > p:
            return t
    print("No threshold found with desired precision and recall balance.")
    return 0.5 

def check_person_split(df_train, df_test, person_col="person_id"):
    train_person_ids = set(df_train[person_col].unique())
    test_person_ids = set(df_test[person_col].unique())
    overlap = train_person_ids.intersection(test_person_ids)
    
    if len(overlap) == 0:
        print("No overlap in person_ids between train and test. The split is correct.")
    else:
        print(f"Warning: Overlap found for {len(overlap)} person_ids. Overlapped IDs: {overlap}")


def check_counts(df_train, df_test, person_col, label_col):
    train_persons_with_1 = df_train.loc[df_train[label_col] == 1, person_col].unique()
    test_persons_with_1 = df_test.loc[df_test[label_col] == 1, person_col].unique()
    
    train_unique_persons = df_train[person_col].nunique()
    test_unique_persons = df_test[person_col].nunique()
    
    print("\n--- Person Counts by Label ---")
    print(f"Train set: {train_unique_persons} unique persons in total.")
    print(f"   {len(train_persons_with_1)} of them have at least one '1'.")
    print(f"Test set:  {test_unique_persons} unique persons in total.")
    print(f"   {len(test_persons_with_1)} of them have at least one '1'.")

def plot_curves(y_test, y_probs, save_dir):
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    
    plt.figure()
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(save_dir, f'precision_recall_curve_best.png'))
    plt.close()
    
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(save_dir, f'roc_curve_best.png'))
    plt.close()

def pipeline(file_path, save_dir):
    print(f"Running file: {file_path}")
    df = pd.read_csv(file_path)
    
    drop_cols = ['order_datetime', 'billing_city', 'shipping_city', 'order_day', 'order_month', 'order_hour', 'order_minute', 'billing_city_frequency', 'shipping_city_frequency']
    if 'repeat_purchase_count' in df.columns:
        drop_cols.append('repeat_purchase_count')
    
    df.drop(columns=drop_cols, inplace=True)
    
    label_col = "repeat_customer"
    person_col = "person_id"
    
    unique_person_ids = df[person_col].unique()
    person_has_one = {pid: 1 if df.loc[df[person_col] == pid, label_col].sum() > 0 else 0 for pid in unique_person_ids}
    person_labels_for_strat = [person_has_one[pid] for pid in unique_person_ids]
    
    train_person_ids, test_person_ids = train_test_split(
        unique_person_ids, test_size=0.2, random_state=42, shuffle=True, stratify=person_labels_for_strat)
    
    df_train = df[df[person_col].isin(train_person_ids)].copy()
    df_test = df[df[person_col].isin(test_person_ids)].copy()
    
    check_person_split(df_train, df_test, person_col)
    check_counts(df_train, df_test, person_col, label_col)
    
    X_train = df_train.drop(columns=[person_col, label_col])
    y_train = df_train[label_col]
    X_test = df_test.drop(columns=[person_col, label_col])
    y_test = df_test[label_col]
    
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print(f"Calculated scale_pos_weight: {scale_pos_weight}")

    model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)

    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    print(f"Training Accuracy (XGBoost): {train_accuracy:.4f}")


    y_prob = model.predict_proba(X_test)[:, 1]
    custom_threshold = find_threshold(y_test, y_prob, min_precision=0.3)
    print(f"\nSelected Threshold: {custom_threshold:.4f}")

    y_pred_custom = (y_prob >= custom_threshold).astype(int)


    accuracy = accuracy_score(y_test, y_pred_custom)
    print("Test Accuracy (XGBoost):", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_custom))

    auc_score = roc_auc_score(y_test, y_prob)
    print("Test AUC (ROC) (XGBoost):", auc_score)

    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nFeature Importances (XGBoost):")
    print(feature_importances)

    plt.figure(figsize=(12, 6))
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances (XGBoost)')
    plt.savefig('feature_importances_xgb_05.png')

    plot_curves(y_test, y_prob, save_dir)


if __name__ == '__main__':
    base_dir = 'xgboost_grid_final_baseline_noncondensed'
    file_paths = {
        'combined/condensed': 'combined_data_condensed.csv',
        'combined/noitemtotal': 'combined_data_woitemtotal.csv'

    }
    
    for subdir, filename in file_paths.items():
        file_path = os.path.join('/nfs/stak/users/aamirj/guille/capstone/encoded-updated', filename)
        save_dir = os.path.join(base_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)
        pipeline(file_path, save_dir)
