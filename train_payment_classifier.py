import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Configuration
PROCESSED_DIR = "processed_data"
TRAIN_FILE = r"C:\Users\karth\hcl-hack\processed_data\train.csv"
TEST_FILE = r"C:\Users\karth\hcl-hack\processed_data\test.csv"
PREPROCESSOR_FILE = r"C:\Users\karth\hcl-hack\processed_data\preprocessor.joblib"
LABEL_ENCODER_FILE = r"C:\Users\karth\hcl-hack\processed_data\label_encoder.joblib"
OUTPUT_MODEL_FILE = "best_model_pipeline.joblib"

def load_data():
    print("Loading data and artifacts...")
    if not all(os.path.exists(f) for f in [TRAIN_FILE, TEST_FILE, PREPROCESSOR_FILE, LABEL_ENCODER_FILE]):
        raise FileNotFoundError("Processed data or artifacts not found. Run preprocess_telecom_data.py first.")
    
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    
    return X_train, y_train, X_test, y_test, preprocessor, le

def train_models(X_train, y_train, X_test, y_test, preprocessor, le):
    # Preprocess data
    print("Transforming data...")
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    models = {}
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
    lr.fit(X_train_proc, y_train)
    models["Logistic Regression"] = lr
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_proc, y_train)
    models["Random Forest"] = rf
    
    # 3. Gradient Boosting (with Tuning)
    print("Tuning Gradient Boosting...")
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1
    )
    gb_grid.fit(X_train_proc, y_train)
    models["Gradient Boosting (Tuned)"] = gb_grid.best_estimator_
    print(f"Best GB Params: {gb_grid.best_params_}")
    
    results = []
    best_model = None
    best_f1 = -1
    best_model_name = ""
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        y_pred = model.predict(X_test_proc)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))
        
        results.append({"model": name, "accuracy": acc, "f1_macro": f1})
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            
    return best_model, best_model_name, best_f1

def save_pipeline(model, preprocessor, filename):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    joblib.dump(pipeline, filename)
    print(f"\nSaved best model pipeline to {filename}")

if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test, preprocessor, le = load_data()
        
        print(f"\nClass Distribution (Train):\n{y_train.value_counts(normalize=True)}")
        print(f"Class Distribution (Test):\n{y_test.value_counts(normalize=True)}")
        
        best_model, best_name, best_f1 = train_models(X_train, y_train, X_test, y_test, preprocessor, le)
        
        print(f"\n🏆 Best Model: {best_name} with Test F1-Macro: {best_f1:.4f}")
        
        save_pipeline(best_model, preprocessor, OUTPUT_MODEL_FILE)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
