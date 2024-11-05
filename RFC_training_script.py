from azureml.core import Workspace, Dataset, Experiment, Run
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt




# Argument parsing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, dest='input_data', help='Input dataset name')
parser.add_argument('--n_estimators', type=int, help='Number of trees in the forest')
parser.add_argument('--max_depth', type=int, help='Maximum depth of the trees')
parser.add_argument('--min_samples_split', type=int, help='Minimum samples required to split')
parser.add_argument('--min_samples_leaf', type=int, help='Minimum samples at a leaf node')
parser.add_argument('--max_features', type=str, help='Number of features to consider')
parser.add_argument('--criterion', type=str, help='Function to measure split quality')
parser.add_argument('--max_samples', type=float, help='Maximum samples for RandomForestClassifier')

args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

# Load the dataset
df = run.input_datasets['row_data'].to_pandas_dataframe()
print("Initial DataFrame shape:", df.shape)

# Drop customerID and convert object columns to category
df.drop(['customerID'], axis=1, inplace=True)
df[df.select_dtypes(include=object).columns] = df.select_dtypes(include=object).astype('category')
print("DataFrame shape after dropping customerID:", df.shape)

# One-hot encoding
df_cleaned = pd.get_dummies(df, drop_first=True)
print("DataFrame shape after one-hot encoding:", df_cleaned.shape)

# Normalize the data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_df = scaler.fit_transform(df_cleaned)
normalized_df = pd.DataFrame(normalized_df, columns=df_cleaned.columns)
print("Normalized DataFrame shape:", normalized_df.shape)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X = normalized_df.drop(['Churn'], axis=1)
y = normalized_df['Churn']
print("Features shape (X):", X.shape)
print("Target shape (y):", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape (X_train):", X_train.shape)
print("Testing set shape (X_test):", X_test.shape)


# Define and train the model
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    min_samples_leaf=args.min_samples_leaf,
    max_features=args.max_features,
    criterion=args.criterion,
    max_samples=args.max_samples
)

rfc.fit(X_train, y_train)
print("Model training completed.")

# Predictions
y_pred = rfc.predict(X_test)

# Scoring 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Model evaluation metrics - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}")

# Logging and submitting run
run.log('accuracy', accuracy)
run.log('recall', recall)
run.log('precision', precision)
run.log('f1_score', f1)


# Save the trained model to outputs directory
import os
import joblib

model_file_path = 'outputs/random_forest_model.pkl'
joblib.dump(rfc, model_file_path)  # Save the model
print(f"Model saved to {model_file_path}")

#---------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
import interpret
import shap
# Initialize Tabular Explainer
print("Initializing Tabular Explainer...")
# Importing TabularExplainer from interpret.ext.blackbox
from interpret.ext.blackbox import TabularExplainer

explainer = TabularExplainer(
                            model=rfc,
                            initialization_examples=X_train,  # Pass the training data
                            features=X_train.columns.tolist(),  # Feature names
                            classes=['No Churn', 'Churn']  # Class names
                            )


# Generate global explanation
print("Generating global explanation...")
global_explanation = explainer.explain_global(X_test)

# Upload global explanation to Azure ML
# Importing interpret client connection
from azureml.interpret import ExplanationClient

# Upload global explanation to Azure ML
client = ExplanationClient.from_run(run)  # Create ExplanationClient from the current run
print('Connection is established')

# Uploading the global explanation object
client.upload_model_explanation(global_explanation, comment='global explanation: all features')

print("Global explanation uploaded successfully.")
