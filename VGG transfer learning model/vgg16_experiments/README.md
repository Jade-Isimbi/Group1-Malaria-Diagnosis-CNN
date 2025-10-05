# VGG16 Transfer Learning Experiments

This folder contains the results of VGG16 transfer learning experiments for malaria diagnosis.

## Experiments Included:
- **exp01_baseline**: Baseline VGG16 model
- **exp02_fine_tuning_4layers**: Fine-tuning with 4 layers (model.h5 removed - too large for GitHub)

## Files in Each Experiment:
- `classification_report.txt`: Detailed classification metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `learning_curves.png`: Training/validation curves
- `model.h5`: Trained model (only included for experiments with files <100MB)
- `roc_curve.png`: ROC curve visualization

## Note:
Some model files (>100MB) were excluded from this repository due to GitHub's file size limit. The models can be regenerated using the training code.

## Results Summary:
Check `final_results.csv` and `final_results_2experiments.csv` for comprehensive performance comparisons.
