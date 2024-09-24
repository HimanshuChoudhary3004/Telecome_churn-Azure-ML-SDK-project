import os
import argparse
import pandas as pd
from azureml.core import Run
from argparse import ArgumentParser


#---------------------------------------------------------------------------------------
# creating argument parser
#--------------------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument('--input',type=str, dest='normalized_data')
parser.add_argument('--output', type=str, dest='split_data')

args = parser.parse_args()


#---------------------------------------------------------------------------------------
# Fetching data and splitting data
#--------------------------------------------------------------------------------------
df = pd.read_csv(os.path.join(args.normalized_data, 'normalized_df.csv'))

X = df.drop(['Churn'], axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Create output directory if it doesn't exist
os.makedirs(args.split_data, exist_ok=True)

# Save the normalized dataframe to the specified output path
# output_path = os.path.join(args.split_data, 'X_train.csv')
# X_train.to_csv(output_path, index=False)

# output_path = os.path.join(args.split_data, 'X_test.csv')
# X_test.to_csv(output_path, index=False)

# output_path = os.path.join(args.split_data, 'y_train.csv')
# y_train.to_csv(output_path, index=False)

# output_path = os.path.join(args.split_data, 'y_test.csv')
# y_test.to_csv(output_path, index=False)

os.makedirs(args.split_data, exist_ok=True)

X_train.to_csv(os.path.join(args.split_data, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(args.split_data, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(args.split_data, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(args.split_data, 'y_test.csv'), index=False)


#---------------------------------------------------------------------------------------
# Logging and submitting run
#--------------------------------------------------------------------------------------

run = Run.get_context()

run.log('X_train', X_train)
run.log('X_test', X_test)
run.log('y_train', y_train)
run.log('y_test', y_test)

run.complete()

