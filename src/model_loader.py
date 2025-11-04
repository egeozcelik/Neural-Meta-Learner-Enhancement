import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


class BaseModelLoader:
    """
    Loads pre-trained CatBoost model and evaluates baseline performance
    """
    
    def __init__(self, model_dir="models/base_model", data_path="data/processed/processed_data.csv"):
        """
        Args:
            model_dir: Directory containing trained model artifacts
            data_path: Path to processed dataset
        """
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
        """Load model, scaler, and feature schema"""
        print("\n" + "=" * 80)
        print("LOADING BASE MODEL ARTIFACTS")
        print("=" * 80)
        
        try:
            # Load model
            model_path = self.model_dir / "best_catboost_model.pkl"
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded: {model_path}")
            
            # Load scaler
            scaler_path = self.model_dir / "robust_scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded: {scaler_path}")
            
            # Load feature columns
            columns_path = self.model_dir / "model_columns.pkl"
            self.feature_columns = joblib.load(columns_path)
            print(f"✓ Feature schema loaded: {len(self.feature_columns)} features")
            
            print("=" * 80)
            return True
            
        except Exception as e:
            print(f"✗ Error loading artifacts: {str(e)}")
            print("=" * 80)
            return False
    
    def load_and_prepare_data(self, test_size=0.2, random_state=42):
        """Load dataset and create train/test split"""
        print("\n" + "=" * 80)
        print("DATA PREPARATION")
        print("=" * 80)
        
        try:
            # Load data
            df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Separate features and target
            if 'winner' not in df.columns:
                raise ValueError("Target column 'winner' not found in dataset")
            
            y = df['winner']
            X = df.drop(['winner'], axis=1)
            
            # Ensure feature alignment
            if not all(col in X.columns for col in self.feature_columns):
                missing = [col for col in self.feature_columns if col not in X.columns]
                raise ValueError(f"Missing features in dataset: {missing[:5]}...")
            
            X = X[self.feature_columns]
            
            # Train/test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"✓ Train set: {self.X_train.shape[0]:,} samples")
            print(f"✓ Test set: {self.X_test.shape[0]:,} samples")
            
            # Scale features
            self.X_train_scaled = self.scaler.transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print(f"✓ Features scaled with RobustScaler")
            
            print("=" * 80)
            return True
            
        except Exception as e:
            print(f"✗ Error preparing data: {str(e)}")
            print("=" * 80)
            return False
    
    def evaluate_baseline(self):
        """Evaluate base model performance"""
        print("\n" + "=" * 80)
        print("BASE MODEL EVALUATION")
        print("=" * 80)
        
        try:
            # Predictions
            y_pred = self.model.predict(self.X_test_scaled)
            y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            self.baseline_metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
            
            # Display results
            print(f"\nModel: CatBoost (Pre-trained)")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            print("\nClassification Report:")
            print("-" * 80)
            target_names = ['Blue Corner', 'Red Corner']
            print(classification_report(self.y_test, y_pred, target_names=target_names))
            
            print("=" * 80)
            return True
            
        except Exception as e:
            print(f"✗ Error during evaluation: {str(e)}")
            print("=" * 80)
            return False
    
    def run_baseline_evaluation(self):
        """Execute complete baseline evaluation pipeline"""
        print("\n" + "=" * 80)
        print("BASELINE EVALUATION PIPELINE")
        print("=" * 80 + "\n")
        
        # Step 1: Load artifacts
        if not self.load_artifacts():
            return None
        
        # Step 2: Prepare data
        if not self.load_and_prepare_data():
            return None
        
        # Step 3: Evaluate
        if not self.evaluate_baseline():
            return None
        
        print("\n" + "=" * 80)
        print("BASELINE EVALUATION COMPLETED")
        print("=" * 80 + "\n")
        
        return self.baseline_metrics
    
    def get_data(self):
        """Return prepared datasets"""
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'X_train_scaled': self.X_train_scaled,
            'X_test_scaled': self.X_test_scaled
        }
    
    def get_model(self):
        """Return base model"""
        return self.model
    
    def get_scaler(self):
        """Return scaler"""
        return self.scaler
    
    def get_baseline_metrics(self):
        """Return baseline metrics"""
        return self.baseline_metrics