import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


class LogisticMetaLearner(BaseEstimator, ClassifierMixin):
    """
    Simple logistic regression for probability calibration
    Fast, interpretable, low-complexity meta-learner
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.name = "Logistic Regression"
    
    def fit(self, X, y):
        """Train meta-learner"""
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced',
            solver='lbfgs'
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)
    
    def get_name(self):
        return self.name


class LightGBMMetaLearner(BaseEstimator, ClassifierMixin):
    """
    LightGBM meta-learner for capturing complex patterns
    Efficient gradient boosting with low memory footprint
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.name = "LightGBM"
    
    def fit(self, X, y):
        """Train meta-learner"""
        self.model = LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            num_leaves=15,
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)
    
    def get_name(self):
        return self.name


class ShallowNeuralNet(nn.Module):
    """
    Shallow neural network architecture for meta-learning
    Lightweight deep learning approach optimized for MacBook Air
    """
    
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.2):
        super(ShallowNeuralNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(16, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.network(x)


class NeuralMetaLearner(BaseEstimator, ClassifierMixin):
    """
    PyTorch neural network meta-learner wrapper
    Handles training, prediction, and probability estimation
    """
    
    def __init__(self, input_dim=None, random_state=42, epochs=50, batch_size=64, learning_rate=0.001):
        self.input_dim = input_dim
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = None
        self.criterion = None
        self.optimizer = None
        
        self.name = "Shallow Neural Network"
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def fit(self, X, y):
        """Train neural network meta-learner"""
        # Initialize model if not already done
        if self.model is None or self.input_dim is None:
            self.input_dim = X.shape[1]
            torch.manual_seed(self.random_state)
            self.model = ShallowNeuralNet(self.input_dim)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.model.to(self.device)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y).to(self.device)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            # Optional: Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"    Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
    
    def get_name(self):
        return self.name


class MetaLearnerFactory:
    """
    Factory class for creating meta-learner instances
    Provides unified interface for all meta-learners
    """
    
    @staticmethod
    def create_meta_learner(learner_type, input_dim=None, random_state=42):
        """
        Create meta-learner instance
        
        Args:
            learner_type: Type of meta-learner ('logistic', 'lightgbm', 'neural')
            input_dim: Input dimension (required for neural network)
            random_state: Random seed for reproducibility
        
        Returns:
            Meta-learner instance
        """
        if learner_type == 'logistic':
            return LogisticMetaLearner(random_state=random_state)
        
        elif learner_type == 'lightgbm':
            return LightGBMMetaLearner(random_state=random_state)
        
        elif learner_type == 'neural':
            if input_dim is None:
                raise ValueError("input_dim must be provided for neural network")
            return NeuralMetaLearner(input_dim=input_dim, random_state=random_state)
        
        else:
            raise ValueError(f"Unknown learner type: {learner_type}")
    
    @staticmethod
    def get_available_learners():
        """Return list of available meta-learner types"""
        return ['logistic', 'lightgbm', 'neural']