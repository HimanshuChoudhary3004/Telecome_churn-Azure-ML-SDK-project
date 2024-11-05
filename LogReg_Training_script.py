import argparse
from azureml.core import Run
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import joblib
import os
from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--input_data',type=str)
parser.add_argument('--penalty', type=str, choices=['l1', 'l2'], required=True, help='Specify the norm used in the penalization.')
parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for stopping criteria.')
parser.add_argument('--C', type=float, required=True, help='Inverse of regularization strength; smaller values specify stronger regularization.')
parser.add_argument('--solver', type=str, choices=['lbfgs', 'liblinear', 'saga', 'newton-cg'], required=True, help='Algorithm to use in the optimization problem.')
parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations to converge.')
parser.add_argument('--l1_ratio', type=float, help='The Elastic Net mixing parameter; only relevant when using `penalty`=`elasticnet`.')

args = parser.parse_args()
print("Parsed arguments:", vars(args))


# Azure ML setup
run = Run.get_context()
ws = run.experiment.workspace

# Load input data
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
scaler = MinMaxScaler()
normalized_df = scaler.fit_transform(df_cleaned)
normalized_df = pd.DataFrame(normalized_df, columns=df_cleaned.columns)
print("Normalized DataFrame shape:", normalized_df.shape)

# Splitting the data into training and testing sets
X = normalized_df.drop(['Churn'], axis=1)
y = normalized_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape (X_train):", X_train.shape)

# Logistic Regression model instantiation
LRC = LogisticRegression(
    penalty=args.penalty,
    solver=args.solver,
    max_iter=args.max_iter,
    C=args.C,
    tol=args.tol,
    random_state=42,
    warm_start=True
)

# Check solver compatibility with L1 penalty
if args.penalty == 'l1' and args.solver not in ['liblinear', 'saga']:
    raise ValueError("L1 penalty requires 'liblinear' or 'saga' solver.")

# Fit the model
LRC.fit(X_train, y_train)
print("Model fitting complete.")

# Make predictions
y_pred = LRC.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(f"Model Evaluation Metrics:\n Accuracy: {accuracy}\n Recall: {recall}\n Precision: {precision}\n F1 Score: {f1}\n AUC: {auc}")

# Log metrics to Azure ML
run.log('accuracy', accuracy)
run.log('recall', recall)
run.log('precision', precision)
run.log('f1', f1)
run.log('auc', auc)
print("Metrics logged to Azure ML.")

run.complete()

# Save the model
os.makedirs('outputs', exist_ok=True)
model_path = os.path.join('outputs', 'Log_Regression_Tele.pkl')
joblib.dump(value=LRC, filename=model_path)
print(f"Model saved to {model_path}")

# Explain the model
tab_explainer = TabularExplainer(model=LRC, initialization_examples=X_train, features=X_train.columns.tolist(), classes=['No Churn', 'Churn'])
global_explanation = tab_explainer.explain_global(X_test)
print("Global explanation generated.")

# Upload the explanation to Azure ML
client = ExplanationClient.from_run(run)
client.upload_model_explanation(global_explanation, comment='Logistic Regression feature importance')
print("Global explanation uploaded successfully.")




