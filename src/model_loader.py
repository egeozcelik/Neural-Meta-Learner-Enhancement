import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


class BaseModelLoader:
    
    def __init__(self, model_dir="models/base_model", data_path="data/processed/processed_data.csv"):
        self.model_dir = Path(model_dir)
        self.data_path = Path(data_path)
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        self.baseline_metrics = {}
    
    def load_artifacts(self):
        print("LOADING BASE MODEL ARTIFACTS")
        print("_" * 80)
        
        try:
            model_path = self.model_dir / "best_catboost_model.pkl"
            self.model = joblib.load(model_path)
            print(f"  ✓ Model loaded from {model_path.name}")
            
            scaler_path = self.model_dir / "robust_scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ Scaler loaded from {scaler_path.name}")
            
            columns_path = self.model_dir / "model_columns.pkl"
            self.feature_columns = joblib.load(columns_path)
            print(f"  ✓ Feature schema loaded: {len(self.feature_columns)} features")
            
            print("_" * 80 + "\n")
            return True
            
        except Exception as e:
            print(f"  ✗ Error loading artifacts: {str(e)}")
            print("_" * 80 + "\n")
            return False
    
    def load_and_prepare_data(self, test_size=0.2, random_state=42):
        print("\n")
        print("DATA PREPARATION")
        print("_" * 80)
        
        try:
            df = pd.read_csv(self.data_path)
            print(f"  ✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            if 'winner' not in df.columns:
                raise ValueError("Target column 'winner' not found in dataset")
            
            y = df['winner']
            X = df.drop(['winner'], axis=1)
            
            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                print(f"  ✓ Encoding {len(cat_cols)} categorical columns...")
                from sklearn.preprocessing import LabelEncoder
                for col in cat_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            if not all(col in X.columns for col in self.feature_columns):
                missing = [col for col in self.feature_columns if col not in X.columns]
                raise ValueError(f"Missing features in dataset: {missing[:5]}...")
            
            X = X[self.feature_columns]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"  ✓ Train set: {self.X_train.shape[0]:,} samples")
            print(f"  ✓ Test set: {self.X_test.shape[0]:,} samples")
            
            self.X_train_scaled = self.scaler.transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print(f"  ✓ Features scaled with RobustScaler")
            
            print("_" * 80 + "\n")
            return True
            
        except Exception as e:
            print(f"  ✗ Error preparing data: {str(e)}")
            print("_" * 80 + "\n")
            return False
    
    def evaluate_baseline(self):
        print("\n")
        print("BASE MODEL EVALUATION")
        print("_" * 80)
        
        try:
            y_pred = self.model.predict(self.X_test_scaled)
            y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            self.baseline_metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
            
            print(f"\n  Model: CatBoost (Pre-trained)")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            
            print("\n  Classification Report:")
            print("  " + "-" * 76)
            target_names = ['Blue Corner', 'Red Corner']
            report = classification_report(self.y_test, y_pred, target_names=target_names)
            for line in report.split('\n'):
                if line.strip():
                    print(f"  {line}")
            
            print("_" * 80 + "\n")
            return True
            
        except Exception as e:
            print(f"  ✗ Error during evaluation: {str(e)}")
            print("_" * 80 + "\n")
            return False
    
    def run_baseline_evaluation(self):
        
        if not self.load_artifacts():
            return None
        
        if not self.load_and_prepare_data():
            return None
        
        if not self.evaluate_baseline():
            return None
        
        print("\n")
        print("BASELINE EVALUATION COMPLETED")
        print("_" * 80)
        
        return self.baseline_metrics
    
    def get_data(self):
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'X_train_scaled': self.X_train_scaled,
            'X_test_scaled': self.X_test_scaled
        }
    
    def get_model(self):
        return self.model
    
    def get_scaler(self):
        return self.scaler
    
    def get_baseline_metrics(self):
        return self.baseline_metrics