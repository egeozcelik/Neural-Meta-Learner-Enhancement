import warnings
warnings.filterwarnings('ignore')

from src.model_loader import BaseModelLoader
loader = BaseModelLoader()
metrics = loader.run_baseline_evaluation()

if metrics:
    print("\nBaseline Metrics Retrieved:")
    print(metrics)



import numpy as np
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

from src.meta_learners import MetaLearnerFactory
for learner_type in ['logistic', 'lightgbm', 'neural']:
    print(f"\nTesting {learner_type}...")
    
    input_dim = X.shape[1] if learner_type == 'neural' else None
    learner = MetaLearnerFactory.create_meta_learner(learner_type, input_dim=input_dim)
    
    learner.fit(X, y)
    predictions = learner.predict(X[:5])
    probabilities = learner.predict_proba(X[:5])
    
    print(f"  ✓ {learner.get_name()} works!")
    print(f"    Predictions: {predictions}")
    print(f"    Probabilities shape: {probabilities.shape}")


from src.ensemble_stacking import EnsembleStacking
data = loader.get_data()
base_model = loader.get_model()
baseline_metrics = loader.get_baseline_metrics()

ensemble = EnsembleStacking(
    base_model=base_model,
    X_train=data['X_train'],
    X_test=data['X_test'],
    y_train=data['y_train'],
    y_test=data['y_test'],
    X_train_scaled=data['X_train_scaled'],
    X_test_scaled=data['X_test_scaled'],
    random_state=42
)

ensemble.create_meta_features(include_original_features=True)

ensemble.train_meta_learners(learner_types=['logistic', 'lightgbm', 'neural'])

comparison_df = ensemble.compare_with_baseline(baseline_metrics)

print("\nEnsemble Stacking Completed.")

from src.evaluation import ModelEvaluation
evaluator = ModelEvaluation(output_dir="results/plots")

evaluator.generate_all_evaluations(
    baseline_model=base_model,
    meta_learners_dict=ensemble.meta_learners,
    y_test=data['y_test'],
    X_test_scaled=data['X_test_scaled'],
    X_test_meta=ensemble.X_test_meta,
    X_train=data['X_train']
)

print("\nAll Evaluations Completed!")