import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.meta_learners import MetaLearnerFactory


class EnsembleStacking:
    """
    Ensemble stacking framework for meta-learning
    Combines base model predictions with meta-learners
    """
    
    def __init__(self, base_model, X_train, X_test, y_train, y_test, 
                 X_train_scaled, X_test_scaled, random_state=42):
        """
        Args:
            base_model: Pre-trained base model (CatBoost)
            X_train: Training features (unscaled)
            X_test: Test features (unscaled)
            y_train: Training labels
            y_test: Test labels
            X_train_scaled: Scaled training features
            X_test_scaled: Scaled test features
            random_state: Random seed
        """
        self.base_model = base_model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.random_state = random_state
        
        self.meta_learners = {}
        self.results = {}
        
        # Meta-features
        self.X_train_meta = None
        self.X_test_meta = None
    
    def create_meta_features(self, include_original_features=True):
        """
        Create meta-feature matrix from base model predictions
        
        Args:
            include_original_features: Whether to include original features
        
        Returns:
            Meta-feature matrices for train and test
        """
        print("\n" + "_" * 80)
        print("META-FEATURE ENGINEERING")
        print("_" * 80)
        
        # Get base model predictions
        train_proba = self.base_model.predict_proba(self.X_train_scaled)
        test_proba = self.base_model.predict_proba(self.X_test_scaled)
        
        print(f"  ✓ Base model probabilities extracted")
        print(f"    Train shape: {train_proba.shape}")
        print(f"    Test shape:  {test_proba.shape}")
        
        if include_original_features:
            # Combine probabilities with original features
            self.X_train_meta = np.hstack([self.X_train_scaled, train_proba])
            self.X_test_meta = np.hstack([self.X_test_scaled, test_proba])
            print(f"  ✓ Original features included")
        else:
            # Use only probabilities
            self.X_train_meta = train_proba
            self.X_test_meta = test_proba
            print(f"  ✓ Using probabilities only")
        
        print(f"  ✓ Meta-features created")
        print(f"    Train meta shape: {self.X_train_meta.shape}")
        print(f"    Test meta shape:  {self.X_test_meta.shape}")
        print("_" * 80 + "\n")
        
        return self.X_train_meta, self.X_test_meta
    
    def train_meta_learners(self, learner_types=['logistic', 'lightgbm', 'neural']):
        """
        Train multiple meta-learners in parallel
        
        Args:
            learner_types: List of meta-learner types to train
        """
        print("_" * 80)
        print("META-LEARNING: ENSEMBLE STACKING")
        print("_" * 80 + "\n")
        
        if self.X_train_meta is None:
            raise ValueError("Meta-features not created. Run create_meta_features() first.")
        
        input_dim = self.X_train_meta.shape[1]
        
        for idx, learner_type in enumerate(learner_types, 1):
            print(f"[{idx}/{len(learner_types)}] Training {learner_type.title()} Meta-Learner")
            print("  " + "-" * 76)
            
            try:
                # Create meta-learner
                meta_learner = MetaLearnerFactory.create_meta_learner(
                    learner_type, 
                    input_dim=input_dim,
                    random_state=self.random_state
                )
                
                # Train meta-learner
                meta_learner.fit(self.X_train_meta, self.y_train)
                print(f"  ✓ Training completed")
                
                # Cross-validation
                print(f"  ✓ Running 5-fold cross-validation...")
                cv_results = self._cross_validate_meta_learner(meta_learner)
                
                # Test set evaluation
                print(f"  ✓ Evaluating on test set...")
                test_metrics = self._evaluate_on_test(meta_learner)
                
                # Store results
                self.meta_learners[learner_type] = meta_learner
                self.results[learner_type] = {
                    'cv_accuracy': cv_results['test_accuracy'].mean(),
                    'cv_f1': cv_results['test_f1'].mean(),
                    'test_accuracy': test_metrics['accuracy'],
                    'test_f1': test_metrics['f1_score'],
                    'test_roc_auc': test_metrics['roc_auc']
                }
                
                print(f"  ✓ Results:")
                print(f"    CV Accuracy: {self.results[learner_type]['cv_accuracy']:.4f}")
                print(f"    CV F1 Score: {self.results[learner_type]['cv_f1']:.4f}")
                print(f"    Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"    Test F1 Score: {test_metrics['f1_score']:.4f}")
                print()
                
            except Exception as e:
                print(f"  ✗ Error training {learner_type}: {str(e)}\n")
        
        print("_" * 80 + "\n")
    
    def _cross_validate_meta_learner(self, meta_learner):
        """
        Perform cross-validation on meta-learner
        
        Args:
            meta_learner: Meta-learner instance
        
        Returns:
            Cross-validation results
        """
        cv_results = cross_validate(
            meta_learner,
            self.X_train_meta,
            self.y_train,
            cv=5,
            scoring=['accuracy', 'f1_weighted'],
            n_jobs=-1,
            return_train_score=False
        )
        
        # Rename f1_weighted to f1 for consistency
        cv_results['test_f1'] = cv_results.pop('test_f1_weighted')
        
        return cv_results
    
    def _evaluate_on_test(self, meta_learner):
        """
        Evaluate meta-learner on test set
        
        Args:
            meta_learner: Meta-learner instance
        
        Returns:
            Test metrics dictionary
        """
        y_pred = meta_learner.predict(self.X_test_meta)
        y_pred_proba = meta_learner.predict_proba(self.X_test_meta)[:, 1]
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
    
    def get_best_meta_learner(self):
        """
        Identify best performing meta-learner based on test F1 score
        
        Returns:
            Tuple of (best_learner_name, best_meta_learner, best_metrics)
        """
        if not self.results:
            raise ValueError("No meta-learners trained yet.")
        
        # Sort by test F1 score
        best_name = max(self.results, key=lambda k: self.results[k]['test_f1'])
        best_learner = self.meta_learners[best_name]
        best_metrics = self.results[best_name]
        
        return best_name, best_learner, best_metrics
    
    def compare_with_baseline(self, baseline_metrics):
        """
        Compare meta-learners with baseline model
        
        Args:
            baseline_metrics: Baseline model metrics dictionary
        
        Returns:
            Comparison DataFrame
        """
        print("_" * 80)
        print("PERFORMANCE COMPARISON")
        print("_" * 80 + "\n")
        
        # Prepare comparison data
        comparison_data = {
            'Model': ['Baseline (CatBoost)'],
            'Accuracy': [baseline_metrics['accuracy']],
            'F1 Score': [baseline_metrics['f1_score']],
            'ROC-AUC': [baseline_metrics['roc_auc']]
        }
        
        for learner_name, metrics in self.results.items():
            comparison_data['Model'].append(f"Meta-Learner: {learner_name.title()}")
            comparison_data['Accuracy'].append(metrics['test_accuracy'])
            comparison_data['F1 Score'].append(metrics['test_f1'])
            comparison_data['ROC-AUC'].append(metrics['test_roc_auc'])
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Calculate improvements
        baseline_f1 = baseline_metrics['f1_score']
        df_comparison['F1 Improvement'] = df_comparison['F1 Score'] - baseline_f1
        df_comparison['F1 Improvement %'] = (df_comparison['F1 Improvement'] / baseline_f1) * 100
        
        # Display results
        print("  Results Summary:")
        print("  " + "-" * 76)
        for idx, row in df_comparison.iterrows():
            print(f"\n  {row['Model']}")
            print(f"    Accuracy:  {row['Accuracy']:.4f}")
            print(f"    F1 Score:  {row['F1 Score']:.4f}")
            print(f"    ROC-AUC:   {row['ROC-AUC']:.4f}")
            if idx > 0:
                improvement = row['F1 Improvement %']
                symbol = "↑" if improvement > 0 else "↓"
                print(f"    Δ F1:      {symbol} {abs(improvement):.2f}%")
        
        print("\n" + "_" * 80 + "\n")
        
        # Find best performer
        best_name, _, best_metrics = self.get_best_meta_learner()
        best_improvement = ((best_metrics['test_f1'] - baseline_f1) / baseline_f1) * 100
        
        print("_" * 80)
        print(f"BEST PERFORMER: {best_name.upper()}")
        print("_" * 80)
        print(f"  Test F1 Score: {best_metrics['test_f1']:.4f}")
        print(f"  Improvement:   ↑ {best_improvement:.2f}%")
        print("_" * 80 + "\n")
        
        return df_comparison
    
    def get_meta_learner(self, learner_type):
        """Get specific meta-learner instance"""
        return self.meta_learners.get(learner_type)
    
    def get_results(self):
        """Get all results"""
        return self.results
    
    def get_meta_features(self):
        """Get meta-feature matrices"""
        return self.X_train_meta, self.X_test_meta