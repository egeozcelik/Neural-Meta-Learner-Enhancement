from src.model_loader import BaseModelLoader

loader = BaseModelLoader()
metrics = loader.run_baseline_evaluation()

if metrics:
    print("\n🎯 Baseline Metrics Retrieved:")
    print(metrics)