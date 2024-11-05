import argparse
from azureml.core import Run
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import joblib
import os
from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--n_estimators', type=int)
parser.add_argument('--max_depth', type=int)
parser.add_argument('--min_samples_split', type=int)
parser.add_argument('--min_samples_leaf', type=int)
parser.add_argument('--max_features', type=str)
parser.add_argument('--criterion', type=str)
parser.add_argument('--max_samples', type=float)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--subsample', type=float)
parser.add_argument('--max_depth_gb', type=int)
parser.add_argument('--min_samples_split_gb', type=int)
parser.add_argument('--min_samples_leaf_gb', type=int)

args = parser.parse_args()

# Get the run context
run = Run.get_context()
ws = run.experiment.workspace
print("Workspace retrieved.")

# Load the dataset
df = run.input_datasets['input_data'].to_pandas_dataframe()
print("Initial DataFrame shape:", df.shape)

# Drop customerID and convert object columns to category
df.drop(['customerID'], axis=1, inplace=True)
object_cols = df.select_dtypes(include='object').columns
df[object_cols] = df[object_cols].astype('category')
X = df.drop(['Churn'], axis=1)
y = df['Churn']

print("Data prepared. Features shape:", X.shape)
print("Target shape:", y.shape)

# One-hot encoding for categorical variables
df_cleaned = pd.get_dummies(X, drop_first=True)
print("One-hot encoding completed. Cleaned DataFrame shape:", df_cleaned.shape)

# Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(df_cleaned)
print("Feature normalization completed.")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify=y)

# Store the feature columns after encoding
X_enc_columns = df_cleaned.columns.tolist()
print("Training set shape (X_train):", X_train.shape)
print("Testing set shape (X_test):", X_test.shape)

# Define individual classifiers
rf_classifier = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    min_samples_leaf=args.min_samples_leaf,
    max_features=args.max_features,
    criterion=args.criterion,
    max_samples=args.max_samples
)
print("Random Forest Classifier defined.")

gb_classifier = GradientBoostingClassifier(
    learning_rate=args.learning_rate,
    subsample=args.subsample,
    max_depth=args.max_depth_gb,
    min_samples_split=args.min_samples_split_gb,
    min_samples_leaf=args.min_samples_leaf_gb
)
print("Gradient Boosting Classifier defined.")

# Create the Voting Classifier
voting_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('gb', gb_classifier)
], voting='soft')
print("Voting Classifier created.")

# Train the Voting Classifier
voting_classifier.fit(X_train, y_train)
print("Voting Classifier training completed.")

# Make predictions
y_pred = voting_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

run.log('accuracy', accuracy)
run.log('recall', recall)
run.log('precision', precision)
run.log('f1_score', f1)

print("Model evaluation metrics logged:")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")

# Save the model and encoded columns
file_path = os.path.join('outputs', 'voting_classifier_model.pkl')
joblib.dump(value=[X_enc_columns, voting_classifier], filename=file_path)
print('Model and encoded columns saved to outputs folder.')

# Explain the model
explainer = TabularExplainer(
    model=voting_classifier,
    initialization_examples=X_train,
    features=X_enc_columns,
    classes=['Not Churn', 'Churn']
)
print("Model explanation initialized.")
# Sample 100 rows from the test set
X_test_sample = X_test[:10]  # or use random sampling

# Explain the model using the sample
global_explanation = explainer.explain_global(X_test_sample)
print("Global explanation created.")

# Upload the explanation to Azure ML
client = ExplanationClient.from_run(run)
client.upload_model_explanation(global_explanation)
print("Model explanation uploaded to Azure ML.")
