import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ===================================
# PATH SETUP
# ===================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data"

MODEL_PATH = BASE_DIR / "models"

VISUALS_PATH = BASE_DIR / "visuals"

MODEL_PATH.mkdir(exist_ok=True)

VISUALS_PATH.mkdir(exist_ok=True)


# ===================================
# LOAD DATA
# ===================================

trader_df = pd.read_csv(
    DATA_PATH / "historical_data.csv"
)

sentiment_df = pd.read_csv(
    DATA_PATH / "fear_greed.csv"
)


# ===================================
# PREPROCESSING
# ===================================

# Convert trader timestamp
trader_df['Timestamp'] = pd.to_datetime(
    trader_df['Timestamp'],
    unit='ms'
)

# Extract date
trader_df['Date'] = trader_df[
    'Timestamp'
].dt.date

# Convert sentiment date
sentiment_df['date'] = pd.to_datetime(
    sentiment_df['date']
).dt.date


# ===================================
# MERGE DATASETS
# ===================================

merged_df = pd.merge(
    trader_df,
    sentiment_df,
    left_on='Date',
    right_on='date',
    how='inner'
)

print("\nMerged Shape:", merged_df.shape)


# ===================================
# CREATE TARGET VARIABLE
# ===================================

merged_df['Win'] = (
    merged_df['Closed PnL'] > 0
).astype(int)


# ===================================
# REMOVE DATA LEAKAGE COLUMNS
# ===================================

drop_columns = [

    # Unique IDs
    'Account',
    'Transaction Hash',
    'Order ID',
    'Trade ID',

    # Time columns
    'Timestamp IST',
    'Timestamp',
    'Date',
    'date',

    # Leakage target
    'Closed PnL',

    # Dominating feature
    'Direction'
]

merged_df = merged_df.drop(
    columns=drop_columns,
    errors='ignore'
)


# ===================================
# SAVE CLEANED DATASET
# ===================================

cleaned_path = DATA_PATH / "cleaned_data.csv"

merged_df.to_csv(
    cleaned_path,
    index=False
)

print("\n✅ Cleaned dataset saved!")


# ===================================
# FEATURES & TARGET
# ===================================

X = merged_df.drop(
    columns=['Win']
)

y = merged_df['Win']


# ===================================
# ENCODE CATEGORICAL FEATURES
# ===================================

categorical_cols = X.select_dtypes(
    include=['object', 'string']
).columns

encoders = {}

for col in categorical_cols:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(
        X[col]
    )

    encoders[col] = encoder


# ===================================
# DATA SPLITTING
# ===================================

# Keep unseen data separate
X_temp, X_unseen, y_temp, y_unseen = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    random_state=42
)


# ===================================
# XGBOOST MODEL
# ===================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)


# ===================================
# MODEL PREDICTIONS
# ===================================

train_pred = model.predict(X_train)

test_pred = model.predict(X_test)

unseen_pred = model.predict(X_unseen)


# ===================================
# MODEL ACCURACY
# ===================================

train_acc = accuracy_score(
    y_train,
    train_pred
)

test_acc = accuracy_score(
    y_test,
    test_pred
)

unseen_acc = accuracy_score(
    y_unseen,
    unseen_pred
)


# ===================================
# RESULTS
# ===================================

print("\n===== XGBOOST RESULTS =====")

print(f"\nTrain Accuracy: {train_acc:.2f}")

print(f"Test Accuracy: {test_acc:.2f}")

print(f"Unseen Accuracy: {unseen_acc:.2f}")

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_unseen,
        unseen_pred
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_unseen,
        unseen_pred
    )
)


# ===================================
# SAVE MODEL & ENCODERS
# ===================================

joblib.dump(
    model,
    MODEL_PATH / "xgboost_model.pkl"
)

joblib.dump(
    encoders,
    MODEL_PATH / "encoders.pkl"
)

print("\n✅ Model Saved Successfully!")


# ===================================
# FEATURE IMPORTANCE
# ===================================

importance = model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====\n")

print(importance_df)


# ===================================
# FEATURE IMPORTANCE GRAPH
# ===================================

# -----------------------------------
# FEATURE IMPORTANCE GRAPH
# -----------------------------------

visuals_path = BASE_DIR / "visuals"

visuals_path.mkdir(exist_ok=True)

plt.figure(figsize=(12, 7))

plt.barh(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xlabel("Importance", fontsize=12)

plt.ylabel("Features", fontsize=12)

plt.title(
    "XGBoost Feature Importance",
    fontsize=16,
    fontweight='bold'
)

plt.gca().invert_yaxis()

plt.tight_layout()

# SAVE GRAPH
graph_path = visuals_path / "feature_importance.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches='tight'
)

# CLOSE GRAPH (IMPORTANT)
plt.close()

print(f"\n✅ Graph saved at: {graph_path}")

print("\n✅ Model Saved Successfully!")