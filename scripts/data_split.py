import argparse
import os
import pandas as pd




# Setting up argument parser

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input',type=str, dest='normalized_data')
parser.add_argument('--output', type=str, dest='split_data')

args = parser.parse_args()


df = pd.read_csv(os.path.join(args.normalized_data, 'normalized_df.csv'))


X = df.drop(['Churn'], axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Create output directory if it doesn't exist
os.makedirs(args.split_data, exist_ok=True)

# Save the normalized dataframe to the specified output path
output_path = os.path.join(args.split_data, 'X_train.csv')
X_train.to_csv(output_path, index=False)

output_path = os.path.join(args.split_data, 'X_test.csv')
X_test.to_csv(output_path, index=False)


output_path = os.path.join(args.split_data, 'y_train.csv')
y_train.to_csv(output_path, index=False)


output_path = os.path.join(args.split_data, 'y_test.csv')
y_test.to_csv(output_path, index=False)

