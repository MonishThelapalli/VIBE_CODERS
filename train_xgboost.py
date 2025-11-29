"""
XGBoost Customer Payment Classifier - Auto Leakage Detection Version

This script trains an XGBoost model while:
1. Automatically detecting leaking features via:
   - Mutual Information
   - Correlation
   - Perfect-predictor checks
2. Removing all target-leaking features (direct + indirect)
3. Excluding sensitive demographic features for fairness
4. Using only safe behavioral, usage, billing, and subscription features

Required packages:
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn scipy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "telecom_customers_full.csv"
OUTPUT_MODEL_FILE = "xgb_customer_payment_classifier.joblib"
CONFUSION_MATRIX_FILE = "confusion_matrix_xgb.png"

# Thresholds for leakage detection
CORR_THRESHOLD = 0.70       # for absolute correlation with target
MI_THRESHOLD = 0.08         # for mutual information with target
PERFECT_PRED_MAX_CLASSES = 1

def load_and_explore_data(filepath):
    """Load data and print basic information"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Handle target column - convert segment_label to numeric if needed
    if 'segment_label' in df.columns and 'target' not in df.columns:
        label_map = {
            'critical': 0,
            'habitual_defaulter': 1,
            'occasional_defaulter': 2
        }
        df['target'] = df['segment_label'].map(label_map)
        print(f"\nCreated target from segment_label:")
        print(f"  critical -> 0")
        print(f"  habitual_defaulter -> 1")
        print(f"  occasional_defaulter -> 2")
    
    if 'target' in df.columns:
        df['target'] = df['target'].astype(int)
        print(f"\nTarget distribution:")
        print(df['target'].value_counts(normalize=True))
    
    return df


def detect_leakage(df, target_col='target',
                   corr_threshold=CORR_THRESHOLD,
                   mi_threshold=MI_THRESHOLD):
    """
    Automatically detect potentially leaking features using:
    - Mutual Information (MI)
    - Correlation (for numeric features)
    - Perfect predictor check (feature -> target mapping is deterministic)
    
    Returns a dict with:
    - mi_scores
    - high_mi_features
    - high_corr_features
    - perfect_predictors
    - auto_leakage_features (union of all above)
    """
    print("\n🔍 Running automatic leakage detection (MI + corr + perfect predictors)...")

    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].values

    # Separate numeric and categorical features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()

    # ---------- Mutual Information (numeric + one-hot categorical) ----------
    X_mi_list = []
    base_feature_for_col = []

    # Numeric as-is
    if numeric_features:
        X_num = df[numeric_features]
        X_mi_list.append(X_num)
        base_feature_for_col.extend(numeric_features)

    # Categorical as one-hot with prefix separator so we can map back
    if categorical_features:
        X_cat = df[categorical_features].astype('object')
        X_cat_dummies = pd.get_dummies(X_cat, prefix_sep="__", drop_first=False)
        X_mi_list.append(X_cat_dummies)
        for col in X_cat_dummies.columns:
            base = col.split("__", 1)[0]  # original column name
            base_feature_for_col.append(base)

    if X_mi_list:
        X_mi = pd.concat(X_mi_list, axis=1)
        print(f"\nNumber of columns used for MI: {X_mi.shape[1]}")
        mi_raw = mutual_info_classif(
            X_mi.values,
            y,
            discrete_features='auto',
            random_state=42
        )

        # Aggregate MI per base feature (take max among its dummies)
        mi_scores = {}
        for col_name, base, mi_val in zip(X_mi.columns, base_feature_for_col, mi_raw):
            if base not in mi_scores:
                mi_scores[base] = mi_val
            else:
                mi_scores[base] = max(mi_scores[base], mi_val)

        # Sort for display
        mi_sorted = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 20 features by Mutual Information with target:")
        for feat, score in mi_sorted[:20]:
            print(f"  {feat:<35} MI = {score:.4f}")

        high_mi_features = [feat for feat, score in mi_sorted if score >= mi_threshold]
        print(f"\nFeatures flagged by MI (>= {mi_threshold}):")
        if high_mi_features:
            for feat in high_mi_features:
                print(f"  ⚠️  {feat} (MI = {mi_scores[feat]:.4f})")
        else:
            print("  ✅ None")
    else:
        mi_scores = {}
        high_mi_features = []
        print("\nNo features available for MI computation (unexpected).")

    # ---------- Correlation (numeric only) ----------
    high_corr_features = []
    if numeric_features:
        print(f"\nChecking numeric features with |corr| >= {corr_threshold}...")
        for col in numeric_features:
            # Some numeric columns can be constant -> corr NaN
            try:
                corr_val = df[col].corr(df[target_col])
                if pd.notnull(corr_val) and abs(corr_val) >= corr_threshold:
                    high_corr_features.append(col)
                    print(f"  ⚠️  {col:<35} |corr| = {abs(corr_val):.4f}")
            except Exception:
                continue
        if not high_corr_features:
            print("  ✅ No high-correlation numeric features found.")
    else:
        print("\nNo numeric features available for correlation check.")

    # ---------- Perfect Predictors ----------
    print("\nChecking for perfect predictors (feature values map to a single target)...")
    perfect_predictors = []
    feature_cols_for_pp = [c for c in df.columns if c != target_col]
    for col in feature_cols_for_pp:
        try:
            # For each feature value, how many distinct targets?
            max_classes_per_value = df.groupby(col)[target_col].nunique().max()
            if max_classes_per_value <= PERFECT_PRED_MAX_CLASSES:
                # Ignore features with too few unique values (like target itself etc.)
                # and ones that are obviously ID-like (we'll handle IDs separately)
                if df[col].nunique() > 1:
                    perfect_predictors.append(col)
                    print(f"  ⚠️  {col} is a perfect predictor (max target classes = {max_classes_per_value})")
        except Exception:
            continue

    if not perfect_predictors:
        print("  ✅ No perfect predictors detected.")

    # ---------- Union for automatic leakage ----------
    auto_leakage_features = sorted(set(high_mi_features) |
                                   set(high_corr_features) |
                                   set(perfect_predictors))

    print(f"\n📌 Auto-detected leakage-related features (union of MI/corr/perfect): {len(auto_leakage_features)}")
    if auto_leakage_features:
        for feat in auto_leakage_features:
            mi_val = mi_scores.get(feat, np.nan)
            print(f"   - {feat} (MI={mi_val:.4f} if available)")
    else:
        print("   ✅ None")

    leakage_info = {
        "mi_scores": mi_scores,
        "high_mi_features": high_mi_features,
        "high_corr_features": high_corr_features,
        "perfect_predictors": perfect_predictors,
        "auto_leakage_features": auto_leakage_features
    }
    return leakage_info


def define_feature_sets(df, leakage_info, target_col="target"):
    """
    Define feature columns with strict leakage prevention and fairness constraints.
    
    Removes:
    1. Target-leaking features (manually known)
    2. Auto-detected leakage features (from MI, corr, perfect predictors)
    3. Sensitive demographic features (for fairness)
    4. ID columns
    """
    # ID columns
    id_cols = ["customer_id", "segment_label"]

    # MANUALLY KNOWN TARGET-LEAKING FEATURES
    manual_leakage_cols = [
        # Direct payment behaviour (we REMOVE all)
        "late_payments_12m",
        "missed_payments_12m",
        "num_payments_late_12m",
        "num_payments_on_time_12m",
        "late_payment_ratio",
        "missed_payment_ratio",
        "avg_payment_delay_days",
        "last_payment_delay_days",
        "max_payment_delay_days_12m",
        "max_days_late",
        "days_overdue_current_bill",
        "is_currently_overdue",
        "last_payment_status",
        # Strong outcome indicators
        "renewal_rate",
        "credit_score_range",
        "loyalty_score",
        "reward_points_balance",
        "wallet_balance",
        # Engineered leakage-type features
        "default_events_12m",
        "delay_severity_score",
        "revenue_per_bill",
        "support_calls_per_bill",
        "complaints_per_support_call",
        "refund_requests_12m",
        "dispute_count_12m",
        "historical_billing_disputes",
        "discounts_or_offers_applied",
        "plan_change_count_last_12m",
    ]

    # SENSITIVE DEMOGRAPHIC FEATURES (for fairness)
    sensitive_cols = [
        "customer_age",
        "customer_region",
        "occupation_type",
        "income_bracket",
        "account_type",
        "num_dependents",
    ]

    # Auto leakage from detection step
    auto_leakage = leakage_info.get("auto_leakage_features", [])

    # Combine all exclusions
    exclude_cols = set(
        [target_col] +
        id_cols +
        manual_leakage_cols +
        sensitive_cols +
        auto_leakage
    )

    # Keep only columns that exist in df
    exclude_cols = [c for c in exclude_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\n📊 Feature Set Summary:")
    print(f"Total columns in data: {df.shape[1]}")
    print(f"Total features used   : {len(feature_cols)}")
    print(f"Excluded (ID)         : {len([c for c in id_cols if c in df.columns])}")
    print(f"Excluded (sensitive)  : {len([c for c in sensitive_cols if c in df.columns])}")
    print(f"Excluded (manual leak): {len([c for c in manual_leakage_cols if c in df.columns])}")
    print(f"Excluded (auto leak)  : {len(auto_leakage)}")
    print(f"\n✅ Safe features used:")
    print(feature_cols)

    return feature_cols, target_col


def split_data(df, feature_cols, target_col):
    """Split data into train and test sets"""
    print("\nSplitting data...")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"\nTrain target distribution:")
    print(y_train.value_counts(normalize=True))
    print(f"\nTest target distribution:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    """Build preprocessing pipeline for numeric and categorical features"""
    print("\nBuilding preprocessor...")

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()

    print(f"Numeric features    : {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_model_pipeline(preprocessor):
    """Build full pipeline with XGBoost"""
    print("\nBuilding XGBoost pipeline...")

    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("xgb", xgb_clf),
    ])

    return model


def tune_hyperparameters(model, X_train, y_train):
    """Perform hyperparameter tuning with cross-validation"""
    print("\n🔧 Tuning hyperparameters (RandomizedSearchCV)...")

    param_distributions = {
        "xgb__n_estimators": randint(100, 400),
        "xgb__max_depth": randint(3, 8),
        "xgb__learning_rate": uniform(0.01, 0.2),
        "xgb__subsample": uniform(0.6, 0.4),
        "xgb__colsample_bytree": uniform(0.6, 0.4),
        "xgb__reg_lambda": uniform(0.0, 2.0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="f1_macro",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42,
    )

    random_search.fit(X_train, y_train)

    print(f"\n✅ Best params: {random_search.best_params_}")
    print(f"✅ Best CV macro F1: {random_search.best_score_:.4f}")

    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n--- Model Evaluation ---")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1:.4f}")

    if acc > 0.98:
        print("\n⚠️  WARNING: Accuracy > 98% - STILL POSSIBLE DATA LEAKAGE.")
        print("    → Inspect the 'Auto-detected leakage features' above and consider excluding more.")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["critical", "habitual_defaulter", "occasional_defaulter"]
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["critical", "habitual", "occasional"],
                yticklabels=["critical", "habitual", "occasional"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - XGBoost (Leakage-Controlled)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    print(f"\n✅ Confusion matrix saved to {CONFUSION_MATRIX_FILE}")

    return acc, f1


def save_model(model, filepath):
    """Save the trained model pipeline"""
    dump(model, filepath)
    print(f"\n✅ Model saved to {filepath}")


if __name__ == "__main__":
    # 1. Load data
    df = load_and_explore_data(INPUT_FILE)

    # 2. Auto-detect leakage via MI + corr + perfect predictors
    leakage_info = detect_leakage(df, target_col='target',
                                  corr_threshold=CORR_THRESHOLD,
                                  mi_threshold=MI_THRESHOLD)

    # 3. Define feature sets (leakage-free + fairness constraints)
    feature_cols, target_col = define_feature_sets(df, leakage_info, target_col="target")

    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, target_col)

    # 5. Build preprocessor
    preprocessor = build_preprocessor(X_train)

    # 6. Build model pipeline
    model = build_model_pipeline(preprocessor)

    # 7. Hyperparameter tuning
    best_model = tune_hyperparameters(model, X_train, y_train)

    # 8. Evaluate on hold-out test set
    test_acc, test_f1 = evaluate_model(best_model, X_test, y_test)

    # 9. Save model
    save_model(best_model, OUTPUT_MODEL_FILE)

    print(f"\n🏆 Final Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Macro F1: {test_f1:.4f}")
    print(f"\n✅ Training complete with automatic leakage control!")

