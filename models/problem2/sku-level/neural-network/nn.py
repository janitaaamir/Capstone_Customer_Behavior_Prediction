import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping


def find_optimal_threshold(y_test, y_pred_prob):

    y_test_flat = y_test.flatten()
    y_pred_prob_flat = y_pred_prob.flatten()

    thresholds = np.linspace(0.01, 0.9, 100)
    f1_scores = []

    for threshold in thresholds:
        y_pred_binary = (y_pred_prob_flat > threshold).astype(int)
        f1 = f1_score(y_test_flat, y_pred_binary, average='macro', zero_division=0)
        f1_scores.append(f1)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n Best Threshold Found: {optimal_threshold:.4f} (Max F1-Score: {f1_scores[optimal_idx]:.4f})")

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, label="F1-Score", color='b')
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold vs F1 Score")
    plt.axvline(x=optimal_threshold, color='r', linestyle="--", label=f"Best Threshold: {optimal_threshold:.2f}")
    plt.legend()
    plt.grid()
    plt.savefig("threshold_optimization_f1.png")
    plt.close()

    print("\n Saved threshold optimization plot!\n")

    return optimal_threshold

def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        weights_tensor = K.constant(weights)
        return K.mean(weights_tensor * K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss


def check_person_split(df_train, df_test, person_col="person_id"):
    train_person_ids = set(df_train[person_col].unique())
    test_person_ids = set(df_test[person_col].unique())
    overlap = train_person_ids.intersection(test_person_ids)

    if len(overlap) == 0:
        print("No overlap in person_ids between train and test. The split is correct.")
    else:
        print(f"Warning: Overlap found for {len(overlap)} person_ids. Overlapped IDs: {overlap}")


def plot_evaluation_curves(y_test, y_pred_prob):
    y_test_flat = y_test.flatten()
    y_pred_prob_flat = y_pred_prob.flatten()

    precision, recall, _ = precision_recall_curve(y_test_flat, y_pred_prob_flat)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig("precision_recall_curve.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test_flat, y_pred_prob_flat)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend()
    plt.grid()
    plt.savefig("roc_auc_curve.png")
    plt.close()


def pipeline():
    file_path = "/model2/encoded.csv"
    df = pd.read_csv(file_path)

    drop_cols = ['order_datetime', 'repeat_purchase_amount', 'billing_city', 'shipping_city',
                 'order_day', 'order_month', 'order_hour', 'order_minute',
                 'billing_city_frequency', 'shipping_city_frequency', 'repeat_customer']

    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    person_col = "person_id"

    all_next_sku_columns = [col for col in df.columns if col.startswith("next_")]

    # with knives
    # unwanted_skus = [
    #     "CK2-1000_count", "EVT-1000_count", "SS0-1000_count",
    #     "KB1-1000_count", "KS0-1000_count", "nan_count",
    #     "SH10-1000_count", "SH8-1000_count", "SH6-1000_count",
    #     "SH4-1000_count", "SHN6-1000_count", "SH10E-1000_count",
    #     "SH6E-1000_count", "SH8E-1000_count", "SHN6E-1000_count",
    #     "SH4E-1000_count", "DS-M_count", "DS-L_count", "EB5-1000_count",
    #     "FFS-1000_count", "DH-1000_count"
    # ]

    # with knives and sheath
    unwanted_skus = [
        "CK2-1000_count", "EVT-1000_count", "SS0-1000_count",
        "KB1-1000_count", "KS0-1000_count", "nan_count",
        "SH10E-1000_count",
        "SH6E-1000_count", "SH8E-1000_count", "SHN6E-1000_count",
        "SH4E-1000_count", "DS-M_count", "DS-L_count", "EB5-1000_count",
        "FFS-1000_count", "DH-1000_count"
    ]

    unwanted_next_skus = ["next_" + sku for sku in unwanted_skus]
    next_sku_columns = [col for col in all_next_sku_columns if col not in unwanted_next_skus]

    df = df.replace({True: 1, False: 0}).infer_objects(copy=False)

    df.fillna(-1, inplace=True)

    y = (df[next_sku_columns] > 0).astype(int)

    columns_to_remove = set(
        [person_col] + all_next_sku_columns + unwanted_skus
    )

    print(columns_to_remove)

    X = df.drop(columns=columns_to_remove, errors="ignore")

    unique_person_ids = df[person_col].unique()
    person_has_purchase = {pid: 1 if (df[df[person_col] == pid][next_sku_columns] > 0).any().any() else 0 for pid in
                           unique_person_ids}
    person_labels_for_strat = [person_has_purchase[pid] for pid in unique_person_ids]

    train_person_ids, test_person_ids = train_test_split(
        unique_person_ids, test_size=0.2, random_state=42, stratify=person_labels_for_strat
    )

    train_idx = df[person_col].isin(train_person_ids)
    test_idx = df[person_col].isin(test_person_ids)

    df_train, df_test = df[train_idx], df[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    check_person_split(df_train, df_test, person_col)

    df_train = df_train.drop(columns=[person_col]).copy()
    df_test = df_test.drop(columns=[person_col]).copy()

    X_train = X_train.to_numpy().astype(np.float32)
    X_test = X_test.to_numpy().astype(np.float32)
    y_train = y_train.to_numpy().astype(np.float32)
    y_test = y_test.to_numpy().astype(np.float32)

    train_mask = y_train.sum(axis=1) > 0
    test_mask = y_test.sum(axis=1) > 0

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    sku_counts = y_train.sum(axis=0)
    total_counts = sku_counts.sum()

    class_weights = total_counts / (len(sku_counts) * sku_counts.clip(min=1))
    # class_weights = (total_counts / (len(sku_counts) * sku_counts.clip(min=1))) ** 0.25
    # class_weights = (total_counts / (len(sku_counts) * sku_counts.clip(min=1))) ** 0.5
    # class_weights = (total_counts / sku_counts.clip(min=1)) ** 0.75

    #
    # model = Sequential([
    #     Input(shape=(X_train.shape[1],)),
    #     Dense(128),
    #     BatchNormalization(),
    #     Activation('relu'),
    #     Dropout(0.2),
    #     Dense(64),
    #     BatchNormalization(),
    #     Activation('relu'),
    #     Dropout(0.1),
    #     Dense(len(next_sku_columns), activation='sigmoid')
    # ])

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),

        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),

        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.1),

        Dense(len(next_sku_columns), activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=weighted_binary_crossentropy(class_weights),
        metrics=[Precision(name="precision"), Recall(name="recall")]
    )


    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )

    y_pred_prob = model.predict(X_test)

    optimal_threshold = find_optimal_threshold(y_test, y_pred_prob)

    y_pred_binary = (y_pred_prob > optimal_threshold).astype(int)

    accuracy = accuracy_score(y_test.flatten(), y_pred_binary.flatten())
    precision = precision_score(y_test.flatten(), y_pred_binary.flatten(), average='macro', zero_division=0)
    recall = recall_score(y_test.flatten(), y_pred_binary.flatten(), average='macro', zero_division=0)
    f1 = f1_score(y_test.flatten(), y_pred_binary.flatten(), average='macro', zero_division=0)
    auc = roc_auc_score(y_test, y_pred_prob, average='weighted')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")

    plot_evaluation_curves(y_test, y_pred_prob)


    sample_indices = np.random.choice(X_test.shape[0], 1, replace=False)
    print("\n Sample Predictions: ")
    for i in sample_indices:
        print(f"Example {i}:")

        true_labels = y_test[i]
        predicted_probs = y_pred_prob[i]
        predicted_labels = (predicted_probs > optimal_threshold).astype(int)

        for sku, true_label, pred_prob, pred_label in zip(next_sku_columns, true_labels, predicted_probs,
                                                          predicted_labels):
            print(
                f"SKU: {sku} | True Label: {true_label} | Predicted Probability: {pred_prob:.4f} | Predicted Label: {pred_label}")

        print("\n")



if __name__ == '__main__':
    pipeline()
