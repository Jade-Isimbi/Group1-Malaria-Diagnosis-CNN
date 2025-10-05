# Baseline CNN Experiments - Results Summary

## Performance Metrics

| Experiment | Accuracy | AUC | Configuration |
|------------|----------|-----|---------------|
| E0 (Baseline b4) | 95.72% | 98.46% | 84×84 images, batch size 4, Adam optimizer |
| E0 (Baseline b64) | 95.30% | 98.49% | 84×84 images, batch size 64, Adam optimizer |
| E2 (Added Conv) | 95.89% | 98.77% | 3 convolutional blocks, no dropout |
| E3 (Dropout) | 95.82% | 98.87% | 3 convolutional blocks, 0.5 dropout |
| E4 (Batch Norm) | [TBD] | [TBD] | Batch normalization added |
| E5 (Data Aug) | [TBD] | [TBD] | Data augmentation techniques |

## Key Findings
- **Best AUC:** E3 (Dropout) with 98.87%
- **Best Accuracy:** E2 (Added Conv) with 95.89%
- **Batch Size Impact:** Smaller batch size (4) performed better than larger (64)
- **Regularization Benefit:** Dropout improved generalization (highest AUC)

## Files Included
- Colab notebook with complete implementation
- Experiment configuration files (config.json)
- Test results with metrics (test_results.json)
- Model weight files excluded due to size constraints (can be regenerated from notebook)

## Repository Structure
```
baseline_cnn_clean/
├── Group1_Malaria_Diagnosis_CNN_Group6_EvenNumber (1).ipynb
├── README.md
└── experiments/
    ├── E0_tfdata_baseline_adam_lr1e-3_b4/
    ├── E0_tfdata_baseline_adam_lr1e-3_b64/
    ├── E2_AddConvLayer/
    ├── E3_Dropout/
    ├── E4_BatchNormalization/
    └── E5_DataAugmentation/
```
