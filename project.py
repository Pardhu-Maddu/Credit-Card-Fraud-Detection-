"""
Credit Card Fraud Detection - MLP with Confusion Matrix Visualizations

Requirements:
    pip install pandas numpy scikit-learn tensorflow imbalanced-learn matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
DATA_PATH = "creditcard.csv"  # change to your local path

df = pd.read_csv(DATA_PATH)
print(df.shape)
print(df["Class"].value_counts(normalize=True))  # class imbalance check

# -----------------------------------------------------------------------
# 2. PREPROCESSING
# -----------------------------------------------------------------------
# 'Amount' and 'Time' are on very different scales than the PCA features (V1-V28)
# so they need scaling. V1-V28 are already PCA components (roughly standardized).
scaler_amount = StandardScaler()
scaler_time = StandardScaler()

df["scaled_amount"] = scaler_amount.fit_transform(df[["Amount"]])
df["scaled_time"] = scaler_time.fit_transform(df[["Time"]])
df = df.drop(["Amount", "Time"], axis=1)

X = df.drop("Class", axis=1).values
y = df["Class"].values

# Stratified split keeps the same fraud ratio in train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4f}")

# -----------------------------------------------------------------------
# 3. HANDLE CLASS IMBALANCE
# -----------------------------------------------------------------------
# Option A (used here): class_weight -- penalize misclassifying fraud more heavily.
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)

# Option B (alternative): oversample fraud cases with SMOTE.
# Uncomment to use instead of / in addition to class_weight.
#
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# -----------------------------------------------------------------------
# 4. BUILD THE MLP
# -----------------------------------------------------------------------
def build_mlp(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(1, activation="sigmoid"),  # binary output: P(fraud)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model

model = build_mlp(X_train.shape[1])
model.summary()

# -----------------------------------------------------------------------
# 5. TRAIN
# -----------------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_pr_auc", mode="max", patience=8, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=2048,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=2,
)

# -----------------------------------------------------------------------
# 6. EVALUATE
# -----------------------------------------------------------------------
y_proba = model.predict(X_test).ravel()

print("\nROC-AUC :", roc_auc_score(y_test, y_proba))
print("PR-AUC  :", average_precision_score(y_test, y_proba))  # more informative than ROC-AUC for imbalance

# Default 0.5 threshold
y_pred_default = (y_proba >= 0.5).astype(int)
print("\n--- Threshold = 0.5 ---")
print(classification_report(y_test, y_pred_default, digits=4))
print(confusion_matrix(y_test, y_pred_default))

# --- Visual confusion matrix (default threshold) ---
cm_default = confusion_matrix(y_test, y_pred_default)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_default, display_labels=['Not Fraud', 'Fraud'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Threshold = 0.5)')
plt.tight_layout()
plt.savefig("confusion_matrix_default.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------
# 7. THRESHOLD TUNING
# -----------------------------------------------------------------------
# In fraud detection, false negatives (missed fraud) and false positives
# (blocking legit transactions) have very different costs. Pick the threshold
# that maximizes F1, or that meets a target recall/precision for your business.
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print(f"\nBest threshold (max F1): {best_threshold:.4f}")
y_pred_best = (y_proba >= best_threshold).astype(int)
print(f"--- Threshold = {best_threshold:.4f} ---")
print(classification_report(y_test, y_pred_best, digits=4))
print(confusion_matrix(y_test, y_pred_best))

# --- Visual confusion matrix (best F1 threshold) ---
cm_best = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=['Not Fraud', 'Fraud'])
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix (Threshold = {best_threshold:.4f})')
plt.tight_layout()
plt.savefig("confusion_matrix_best.png", dpi=150)
plt.show()

# --- Side-by-side comparison of both thresholds ---
fig_cm, axes_cm = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(cm_default, display_labels=['Not Fraud', 'Fraud']).plot(
    ax=axes_cm[0], cmap='Blues', values_format='d', colorbar=False)
axes_cm[0].set_title('Threshold = 0.5')

ConfusionMatrixDisplay(cm_best, display_labels=['Not Fraud', 'Fraud']).plot(
    ax=axes_cm[1], cmap='Blues', values_format='d', colorbar=False)
axes_cm[1].set_title(f'Threshold = {best_threshold:.4f}')

plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------
# 8. PLOTS
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Precision-Recall curve
axes[0].plot(recalls, precisions)
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr)
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")

# Training history (loss)
axes[2].plot(history.history["loss"], label="train loss")
axes[2].plot(history.history["val_loss"], label="val loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")
axes[2].set_title("Training History")
axes[2].legend()

plt.tight_layout()
plt.savefig("fraud_detection_results.png", dpi=150)
print("\nSaved plots to fraud_detection_results.png")

# -----------------------------------------------------------------------
# 9. SAVE MODEL
# -----------------------------------------------------------------------
model.save("fraud_detection_mlp.keras")
print("Model saved to fraud_detection_mlp.keras")