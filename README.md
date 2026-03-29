# Prediction of Concrete Compressive Strength

## Project Overview
This project predicts concrete compressive strength (MPa) from concrete mix design
features using regression models.

The final workflow includes:
- Data preprocessing and exploratory analysis
- Outlier handling and duplicate removal
- Feature scaling
- Linear Regression modeling
- SVM (SVR) modeling
- Model evaluation with MSE, RMSE, MAE, and R2

## Canonical Project Files
Use these as the primary project artifacts:

- `Concrete Compressive Strength/Concrete_Strength_Report_Ammaar_Shimrin.ipynb` (main notebook)
- `Concrete Compressive Strength/Concrete_Data.csv` (dataset)

Submission exports are available in:
- `Concrete Compressive Strength/Concrete_Strength_Report_Ammaar_Shimrin.pdf`
- `Concrete Compressive Strength/Concrete_Strength_Report_Ammaar_Shimrin.html`

## Cleanup Notes
Non-core and duplicate files were removed from the active workspace to reduce
confusion. Archived duplicates retained in:

- `_archive_non_project_2026-03-29/`

## Recommended Next Improvements
1. Add cross-validation for both models.
2. Add SVR hyperparameter tuning (GridSearchCV).
3. Add a short reproducibility cell (random seeds + package versions).
4. Add feature importance or sensitivity discussion for interpretability.

## Dataset Source
UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength
