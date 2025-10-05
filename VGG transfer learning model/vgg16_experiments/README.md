# VGG16 Transfer Learning Experiments

This folder contains the results of VGG16 transfer learning experiments for malaria diagnosis.

## Experiments Included:
- **exp01_baseline**: Baseline VGG16 model - Standard VGG16 architecture with transfer learning
- **exp02_fine_tuning_4layers**: Fine-tuning with 4 layers - Custom fine-tuning approach for improved performance

## Files in Each Experiment:
- `classification_report.txt`: Detailed classification metrics including precision, recall, F1-score
- `confusion_matrix.png`: Confusion matrix visualization showing true vs predicted labels
- `learning_curves.png`: Training/validation accuracy and loss curves
- `roc_curve.png`: ROC curve visualization with AUC scores

## Results Summary:
Check `final_results_2experiments.csv` for comprehensive performance comparison between the baseline and fine-tuning experiments.

## Model Files:
Model files (.h5) were excluded due to GitHub's 100MB file size limit. The trained models can be regenerated using the provided notebook code.

## Performance:
Both experiments show promising results for malaria diagnosis using VGG16 transfer learning, with the fine-tuning approach demonstrating improved performance over the baseline model.
