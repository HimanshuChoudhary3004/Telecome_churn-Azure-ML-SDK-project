# Telecom Customer Churn Prediction using Azure Machine Learning

This project aims to predict customer churn in a telecom company using Azure Machine Learning. The solution explores different classification algorithms to identify the most effective model, optimize its hyperparameters, and deploy it for analysis. The primary objective is to maximize normalized recall.

## Project Overview

The project follows these steps:

1. **AutoML Evaluation**:
   - Created an AzureML AutoML job to evaluate various classification algorithms.
   - Focused on optimizing **normalized recall** as the primary metric.
   - Analyzed the performance of algorithms by plotting their **accuracy**, **recall**, and **AUC** using a line chart.

2. **Model Selection**:
   - Identified the following models as top performers:
     - Random Forest (RFC)
     - Voting Random Forest
     - Stochastic Gradient Descent (SGD)
     - Logistic Regression
     - Stacked Random Forest
   - Proceeded with Random Forest Classifier (RFC) training in the next phase.

3. **Training Script**:
   - Developed a training script (`train_model.py`) that:
     - Fetches data from an Azure Storage account.
     - Performs preprocessing, normalization, and splitting of the dataset.
     - Trains the selected model (RFC).
   
4. **Hyperparameter Tuning**:
   - Used Azure HyperDrive for hyperparameter tuning, employing **Bayesian Optimization** for sampling.
   - Tuned hyperparameters included:
     - `n_estimators`
     - `max_depth`
     - `min_samples_split`
     - `min_samples_leaf`
     - `max_features`
     - `criterion`

5. **Model Interpretation and Registration**:
   - Selected the best run from HyperDrive.
   - Logged model explanations using the **interpret** package.
   - Registered the final model in the AzureML workspace.

## Project Structure



Best hyperparameters are:

 --input_data : DatasetConsumptionConfig:row_data

 --penalty : l1

 --tol : 0.012307751027352702

 --C : 2.412974474350091

 --solver : liblinear

 --max_iter : 200

 --l1_ratio : 0.8940615165615252
Best run metrics are:
accuracy: 0.7896233120113717
recall: 0.5240641711229946
precision: 0.6242038216560509
auc: 0.704916887110384
f1: 0.5697674418604651
