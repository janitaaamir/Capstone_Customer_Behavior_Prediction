import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             precision_recall_curve, roc_curve, average_precision_score)
from sklearn.metrics import confusion_matrix


plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


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


def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    best_f1 = f1_scores[best_index]

    print(f"Optimal Threshold based on Precision-Recall Curve: {best_threshold:.4f}")
    print(f"Best F1-score at this threshold: {best_f1:.4f}")

    return best_threshold


def plot_curves(y_test, y_probs, save_dir):
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', markersize=8, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve_best.png'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', markersize=8, linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(save_dir, 'roc_curve_best.png'), bbox_inches='tight')
    plt.close()



def pipeline(file_path, save_dir):
    print(f"Running file: {file_path}")
    df = pd.read_csv(file_path)

    drop_cols = ['order_datetime', 'billing_city', 'shipping_city', 'order_day', 'order_month', 'order_hour',
                 'order_minute', 'billing_city_frequency', 'shipping_city_frequency']
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

    param_grid = {
        'max_depth': [20, 50, None],
        'min_samples_leaf': [2, 3, 5],
        'max_features': ['sqrt'],
        'n_estimators': [100, 200, 500],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True]
    }

    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)

    print("\nAll Parameter Combinations Tested")
    cv_results = grid_search.cv_results_
    for i in range(len(cv_results["params"])):
        print(f"Combination {i+1}: {cv_results['params'][i]}")
        print(f"Mean AP Score: {cv_results['mean_test_score'][i]:.5f}  |  Std: {cv_results['std_test_score'][i]:.5f}\n")

    grid_results_file = os.path.join(save_dir, "grid_search_results.txt")
    with open(grid_results_file, "w") as f:
        f.write("All Parameter Combinations Tested \n")
        for i in range(len(cv_results["params"])):
            f.write(f"Combination {i+1}: {cv_results['params'][i]}\n")
            f.write(f"Mean AP Score: {cv_results['mean_test_score'][i]:.5f}  |  Std: {cv_results['std_test_score'][i]:.5f}\n\n")

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
    print(f"Train Accuracy: {train_accuracy:.4f}")

    y_prob = best_model.predict_proba(X_test)[:, 1]
    best_threshold = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_threshold).astype(int)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)

    feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    print("Best parameters found:", best_params)
    print(f"Best Threshold (F1-optimized): {best_threshold:.4f}")
    print("Test Accuracy:", test_accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Test AUC (ROC):", test_auc)
    print("\nFeature Importances:")
    print(feature_importances)

    results_file = os.path.join(save_dir, "best_model_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best Threshold: {best_threshold:.4f}\n")
        f.write(f"Train Accuracy: {train_accuracy:.5f}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"Test Accuracy: {test_accuracy:.5f}\n")
        f.write(f"Test AUC (ROC): {test_auc:.5f}\n")
        f.write("Feature Importances:\n")
        f.write(feature_importances.to_string())

    plot_curves(y_test, y_prob, save_dir)

    print(f"\nResults saved in: {results_file}")


if __name__ == '__main__':
    base_dir = 'pb1_rf_final'
    file_paths = {
        'baseline/condensed': 'combined_data_condensed.csv',
        'baseline/noitemtotal': 'combined_data_woitemtotal.csv'
    }

    for subdir, filename in file_paths.items():
        file_path = os.path.join('/Users/janitaaamir/Downloads/non-encoded/encoded', filename)
        save_dir = os.path.join(base_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)
        pipeline(file_path, save_dir)

