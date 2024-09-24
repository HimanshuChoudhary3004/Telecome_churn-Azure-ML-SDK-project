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

args = parser.parse_args()


#---------------------------------------------------------------------------------------
# Fetching data and training model for scoring
#--------------------------------------------------------------------------------------
# X_train = pd.read_csv(os.path.join(args.split_data, 'X_train.csv'))
# y_train = pd.read_csv(os.path.join(args.split_data, 'y_train.csv'))
# X_test = pd.read_csv(os.path.join(args.split_data, 'X_test.csv'))
# y_test = pd.read_csv(os.path.join(args.split_data, 'y_test.csv'))

X_train = pd.read_csv(os.path.join(args.split_data, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(args.split_data, 'y_train.csv')).values.ravel()
X_test = pd.read_csv(os.path.join(args.split_data, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(args.split_data, 'y_test.csv')).values.ravel()


from sklearn.ensemble import RandomForestClassifier

rfc =  RandomForestClassifier(n_estimators=100,
                             max_depth=5,
                             )

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

# Scoring 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

accuracy =accuracy_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred,)
f1 = f1_score(y_test,y_pred)



#---------------------------------------------------------------------------------------
# Logging and submitting run
#--------------------------------------------------------------------------------------
run = Run.get_context()

run.log('accuracy :',accuracy)
run.log('recall_score :',recall)
run.log('recall_score :',precision)
run.log('f1_score :',f1)

run.complete()
