import os
import argparse
import pandas as pd 
from azureml.core import Run
from argparse import ArgumentParser

#---------------------------------------------------------------------------------------
# creating argument parser
#--------------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('--input', type=str, dest='split_data')
parser.add_argument('--output', type=str, dest='model_output_data')
# Add hyperparameters
parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest')
parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the trees')
parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum samples required to split')
parser.add_argument('--min_samples_leaf', type=int, default=1, help='Minimum samples at a leaf node')
parser.add_argument('--max_features', type=str, default='auto', help='Number of features to consider')
parser.add_argument('--criterion', type=str, default='gini', help='Function to measure split quality')

args = parser.parse_args()

run = Run.get_context()
#---------------------------------------------------------------------------------------
# Fetching data and training model for scoring
#--------------------------------------------------------------------------------------

X_train = pd.read_csv(os.path.join(args.split_data, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(args.split_data, 'y_train.csv')).values.ravel()
X_test = pd.read_csv(os.path.join(args.split_data, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(args.split_data, 'y_test.csv')).values.ravel()


from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    min_samples_leaf=args.min_samples_leaf,
    max_features=args.max_features,
    criterion=args.criterion
)


rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

# Scoring 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

accuracy = accuracy_score(y_test,y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


#---------------------------------------------------------------------------------------
# Logging and submitting run
#--------------------------------------------------------------------------------------


run.log('accuracy', accuracy)
run.log('recall', recall)
run.log('precision', precision)
run.log('f1_score', f1)
print('accuracy logging done')
 
run.complete()

