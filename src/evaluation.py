import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc


class ModelEvaluation:
    """
    Comprehensive evaluation and visualization module
    Generates confusion matrices, ROC curves, and feature importance plots
    """
    
    def __init__(self, output_dir="results/plots"):
        """
        Args:
            output_dir: Directory to save visualization plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, class_names=None):
        """
        Plot confusion matrix for a model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            class_names: List of class names
        """
        if class_names is None:
            class_names = ['Blue Corner', 'Red Corner']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Confusion matrix saved: {filepath}")
        plt.close()
    
    def plot_roc_curves(self, models_dict, y_test, X_test_meta, baseline_model=None, 
                       X_test_scaled=None):
        """
        Plot ROC curves for all models on the same plot
        
        Args:
            models_dict: Dictionary of {model_name: meta_learner}
            y_test: True labels
            X_test_meta: Test meta-features
            baseline_model: Base CatBoost model (optional)
            X_test_scaled: Scaled test features for baseline (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot baseline if provided
        if baseline_model is not None and X_test_scaled is not None:
            y_proba_base = baseline_model.predict_proba(X_test_scaled)[:, 1]
            fpr_base, tpr_base, _ = roc_curve(y_test, y_proba_base)
            roc_auc_base = auc(fpr_base, tpr_base)
            
            ax.plot(fpr_base, tpr_base, linewidth=2.5, linestyle='--',
                   label=f'Baseline (CatBoost) - AUC = {roc_auc_base:.4f}',
                   color='gray', alpha=0.7)
        
        # Color palette for meta-learners
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # Plot meta-learners
        for idx, (model_name, meta_learner) in enumerate(models_dict.items()):
            y_proba = meta_learner.predict_proba(X_test_meta)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2.5,
                   label=f'{model_name.title()} - AUC = {roc_auc:.4f}',
                   color=colors[idx % len(colors)])
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.4, label='Random Classifier')
        
        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves: Meta-Learners vs Baseline', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / "roc_curves_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ ROC curves saved: {filepath}")
        plt.close()
    
    def plot_feature_importance(self, base_model, meta_learner, X_train, 
                               model_name, top_n=20):
        """
        Compare feature importance between base model and meta-learner
        
        Args:
            base_model: Base CatBoost model
            meta_learner: Trained meta-learner (must support feature_importances_)
            X_train: Training features
            model_name: Name of meta-learner
            top_n: Number of top features to display
        """
        # Check if meta-learner has feature_importances_
        if not hasattr(meta_learner, 'model') or not hasattr(meta_learner.model, 'feature_importances_'):
            print(f"  ⚠ {model_name} does not support feature importance visualization")
            return
        
        # Get base model feature importance
        base_importance = base_model.feature_importances_
        base_features = X_train.columns.tolist()
        
        # Get meta-learner feature importance
        # Meta-features = [original features + base model probabilities]
        meta_importance = meta_learner.model.feature_importances_
        meta_features = base_features + ['Base_Prob_Class0', 'Base_Prob_Class1']
        
        # Create DataFrames
        df_base = pd.DataFrame({
            'Feature': base_features,
            'Importance': base_importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        df_meta = pd.DataFrame({
            'Feature': meta_features,
            'Importance': meta_importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot base model
        sns.barplot(data=df_base, y='Feature', x='Importance', 
                   palette='viridis', ax=ax1)
        ax1.set_title('Base Model (CatBoost)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Feature', fontsize=12, fontweight='bold')
        
        # Plot meta-learner
        # Highlight base probability features
        colors = ['#F18F01' if 'Base_Prob' in feat else '#2E86AB' 
                  for feat in df_meta['Feature']]
        
        sns.barplot(data=df_meta, y='Feature', x='Importance', 
                   palette=colors, ax=ax2)
        ax2.set_title(f'Meta-Learner ({model_name.title()})', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Feature', fontsize=12, fontweight='bold')
        
        # Add legend for meta-learner
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Original Features'),
            Patch(facecolor='#F18F01', label='Base Model Probabilities')
        ]
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        # Save
        filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Feature importance saved: {filepath}")
        plt.close()
    
    def generate_all_evaluations(self, baseline_model, meta_learners_dict, 
                                y_test, X_test_scaled, X_test_meta, X_train):
        """
        Generate all evaluation plots in one go
        
        Args:
            baseline_model: Base CatBoost model
            meta_learners_dict: Dictionary of {model_name: meta_learner}
            y_test: True test labels
            X_test_scaled: Scaled test features (for baseline)
            X_test_meta: Meta-features (for meta-learners)
            X_train: Training features (for feature names)
        """
        print("\n")
        print("EVALUATION & VISUALIZATION")
        print("_" * 80 + "\n")
        
        # 1. Confusion Matrices
        print("  [1/3] Generating confusion matrices...")
        
        # Baseline confusion matrix
        y_pred_base = baseline_model.predict(X_test_scaled)
        self.plot_confusion_matrix(y_test, y_pred_base, "Baseline (CatBoost)")
        
        # Meta-learners confusion matrices
        for model_name, meta_learner in meta_learners_dict.items():
            y_pred_meta = meta_learner.predict(X_test_meta)
            self.plot_confusion_matrix(y_test, y_pred_meta, f"Meta-Learner ({model_name.title()})")
        
        # 2. ROC Curves
        print("\n  [2/3] Generating ROC curves...")
        self.plot_roc_curves(meta_learners_dict, y_test, X_test_meta, 
                            baseline_model, X_test_scaled)
        
        # 3. Feature Importance
        print("\n  [3/3] Generating feature importance plots...")
        for model_name, meta_learner in meta_learners_dict.items():
            self.plot_feature_importance(baseline_model, meta_learner, X_train, model_name)
        
        print("\n")
        print("EVALUATION COMPLETED")
        print("\n")
        print(f"  All plots saved in: {self.output_dir.absolute()}")
        print("_" * 80 + "\n")