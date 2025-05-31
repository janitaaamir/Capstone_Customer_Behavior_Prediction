import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, roc_curve
import joblib

def find_best_threshold_f1_favor_recall(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = np.argmax(f1_scores + 0.01 * recall)
    return thresholds[best_idx], precision[best_idx], recall[best_idx], f1_scores[best_idx]


plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


def load_data(file_path):
    df = pd.read_csv(file_path)
    drop_cols = ['order_datetime', 'repeat_purchase_amount', 'billing_city', 'shipping_city',
                 'order_day', 'order_month', 'order_hour', 'order_minute',
                 'billing_city_frequency', 'shipping_city_frequency', 'repeat_customer']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # with knives
    unwanted_skus = [
        "CK2-1000_count", "EVT-1000_count", "SS0-1000_count",
        "KB1-1000_count", "KS0-1000_count", "nan_count",
        "SH10-1000_count", "SH8-1000_count", "SH6-1000_count",
        "SH4-1000_count", "SHN6-1000_count", "SH10E-1000_count",
        "SH6E-1000_count", "SH8E-1000_count", "SHN6E-1000_count",
        "SH4E-1000_count", "DS-M_count", "DS-L_count", "EB5-1000_count",
        "FFS-1000_count", "DH-1000_count"
    ]

    # # with knives + sheath
    # unwanted_skus = [
    #     "CK2-1000_count", "EVT-1000_count", "SS0-1000_count",
    #     "KB1-1000_count", "KS0-1000_count", "nan_count",
    #     "SH10E-1000_count", "SH6E-1000_count", "SH8E-1000_count", "SHN6E-1000_count",
    #     "SH4E-1000_count", "DS-M_count", "DS-L_count", "EB5-1000_count",
    #     "FFS-1000_count", "DH-1000_count"
    # ]

    unwanted_next_skus = ["next_" + sku for sku in unwanted_skus]
    all_next_sku_columns = [col for col in df.columns if col.startswith("next_")]
    next_sku_columns = [col for col in all_next_sku_columns if col not in unwanted_next_skus]

    most_frequent_cols = [col for col in df.columns if col.startswith("most_frequent_product_")]
    df.drop(columns=most_frequent_cols, inplace=True, errors='ignore')

    df = df.replace({True: 1, False: 0}).infer_objects(copy=False)
    df.fillna(-1, inplace=True)
    return df, next_sku_columns, unwanted_skus

def train_random_forest(X_train, y_train, **kwargs):
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)
    model.fit(X_train, y_train)
    return model


def plot_evaluation_curves(y_test, y_pred_prob, sku):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {sku}')
    plt.legend()
    plt.grid()
    plt.savefig(f'figures/precision_recall_curve_{sku}.png')
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-AUC Curve - {sku}')
    plt.legend()
    plt.grid()
    plt.savefig(f'figures/roc_auc_curve_{sku}.png')
    plt.close()

def pipeline(file_path):
    df, next_sku_columns, unwanted_skus = load_data(file_path)
    person_col = "person_id"
    X = df.drop(columns=[person_col] + next_sku_columns + unwanted_skus, errors="ignore")

    next_sku_count_cols = [col for col in X.columns if col.startswith("next_") and col.endswith("_count")]
    X.drop(columns=next_sku_count_cols, inplace=True, errors="ignore")

    feature_names = X.columns.tolist()
    unique_person_ids = df[person_col].unique()
    train_person_ids, test_person_ids = train_test_split(
        unique_person_ids, test_size=0.2, random_state=42,
        stratify=[1 if (df[df[person_col] == pid][next_sku_columns] > 0).any().any() else 0 for pid in
                  unique_person_ids]
    )
    train_idx = df[person_col].isin(train_person_ids)
    test_idx = df[person_col].isin(test_person_ids)
    X_train, X_test = X[train_idx], X[test_idx]
    X_train, X_test = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32)

    best_models = {}
    all_y_train = []
    all_y_test = []
    all_y_pred_train = []
    all_y_pred_test = []
    all_y_pred_prob = []
    overall_feature_importance = np.zeros(X_train.shape[1])

    individual_auc_scores = {}

    for sku in next_sku_columns:
        print(f'\nTraining Random Forest for SKU: {sku}')
        y_train, y_test = (df.loc[train_idx, sku] > 0).astype(int), (df.loc[test_idx, sku] > 0).astype(int)
        y_train, y_test = y_train.to_numpy().astype(int), y_test.to_numpy().astype(int)
        all_y_train.append(y_train)
        all_y_test.append(y_test)

        # param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }


        grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
                                   param_grid, scoring='roc_auc', cv=3)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f'Best Params: {best_params}')

        model = train_random_forest(X_train, y_train, **best_params)

        # model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=None)

        best_models[sku] = model

        joblib.dump(model, f'randomforest_models/random_forest_{sku}.joblib')

        y_pred_train = model.predict(X_train)

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        threshold, precision_val, recall_val, f1_val = find_best_threshold_f1_favor_recall(y_test, y_pred_prob)
        y_pred_test = (y_pred_prob >= threshold).astype(int)

        print(
            f"Selected threshold for {sku}: {threshold:.4f} | Precision: {precision_val:.4f} | Recall: {recall_val:.4f} | F1: {f1_val:.4f}")


        auc = roc_auc_score(y_test, y_pred_prob)
        individual_auc_scores[sku] = auc

        all_y_pred_train.append(y_pred_train)
        all_y_pred_test.append(y_pred_test)
        all_y_pred_prob.append(y_pred_prob)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f'Training Accuracy for {sku}: {train_accuracy:.4f}')
        print(f'Testing Accuracy for {sku}: {test_accuracy:.4f}')

        feature_importances = model.feature_importances_
        overall_feature_importance += feature_importances

        sorted_indices = np.argsort(feature_importances)[::-1]
        print(f"Top 10 Important Features for {sku}:")
        for i in range(10):
            print(f"{i + 1}. {feature_names[sorted_indices[i]]}: {feature_importances[sorted_indices[i]]:.4f}")

        plot_evaluation_curves(y_test, y_pred_prob, sku)

    all_y_train = np.concatenate(all_y_train)
    all_y_test = np.concatenate(all_y_test)
    all_y_pred_train = np.concatenate(all_y_pred_train)
    all_y_pred_test = np.concatenate(all_y_pred_test)
    all_y_pred_prob = np.concatenate(all_y_pred_prob)

    overall_train_accuracy = accuracy_score(all_y_train, all_y_pred_train)
    overall_test_accuracy = accuracy_score(all_y_test, all_y_pred_test)

    overall_precision = precision_score(all_y_test, all_y_pred_test, average='macro', zero_division=0)
    overall_recall = recall_score(all_y_test, all_y_pred_test, average='macro', zero_division=0)
    overall_f1 = f1_score(all_y_test, all_y_pred_test, average='macro', zero_division=0)
    overall_auc = roc_auc_score(all_y_test, all_y_pred_prob)

    overall_neg_precision = precision_score(all_y_test, all_y_pred_test, pos_label=0, zero_division=0)
    overall_neg_recall = recall_score(all_y_test, all_y_pred_test, pos_label=0, zero_division=0)


    precision, recall, _ = precision_recall_curve(all_y_test, all_y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', markersize=8, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Micro-Averaged Precision-Recall Curve')
    plt.grid()
    plt.savefig('figures2/Item Level/random_forest_baseline_precision_recall_curve.pdf', bbox_inches='tight')
    plt.close()

    fpr, tpr, _ = roc_curve(all_y_test, all_y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', markersize=8, linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Averaged ROC Curve')
    plt.grid()
    plt.savefig('figures2/Item Level/random_forest_baseline_roc_curve.pdf', bbox_inches='tight')
    plt.close()


    print('\nOverall Model Performance: ')
    print(f'Overall Training Accuracy: {overall_train_accuracy:.4f}')
    print(f'Overall Testing Accuracy: {overall_test_accuracy:.4f}')
    print(f'Overall Precision: {overall_precision:.4f}')
    print(f'Overall Recall: {overall_recall:.4f}')
    print(f'Overall F1 Score: {overall_f1:.4f}')
    print(f'Overall AUC Score: {overall_auc:.4f}')

    print(f'Overall Negative Precision: {overall_neg_precision:.4f}')
    print(f'Overall Negative Recall: {overall_neg_recall:.4f}')


    print('\nSaved all trained models and evaluation plots!')

    print('\nIndividual AUC Scores per SKU:')
    for sku, auc in individual_auc_scores.items():
        print(f'{sku}: {auc:.4f}')


    return best_models

if __name__ == '__main__':
    file_path = '/Users/janitaaamir/PycharmProjects/Capstone/model2/encoded.csv'
    best_models = pipeline(file_path)
