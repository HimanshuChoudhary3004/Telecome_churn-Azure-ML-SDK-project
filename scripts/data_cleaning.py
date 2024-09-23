from azureml.core import Workspace, Dataset, Experiment, Run
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input_data', help='Input dataset name')
    parser.add_argument('--output', type=str, dest='cleaned_data', help='Output dataset path')
    
    args = parser.parse_args()

    run = Run.get_context()

    ws = run.experiment.workspace

    df = run.input_datasets['row_data'].to_pandas_dataframe()
   

    df.drop(['customerID'],axis=1,inplace = True)

    df[df.select_dtypes(include=object).columns]=df.select_dtypes(include=object).astype('category')
    df['Churn'].apply(lambda x : 1 if x=='yes' else 0)
    df_cleaned = pd.get_dummies(df,drop_first=True)
    df_cleaned[df_cleaned.select_dtypes(include=bool).columns] = df_cleaned[df_cleaned.select_dtypes(include=bool).columns].applymap(lambda x: 1 if x== True else 0)


    feat_importance = abs(df_cleaned.corr()['Churn']).sort_values(ascending=False)[1:]

    # saving output locally
    local_path = './outputs/df_cleaned.csv'
    os.makedirs('./outputs',exist_ok=True)
    df_cleaned.to_csv(local_path, index=False)

    run.log("Feature Importance", feat_importance.to_dict())

    # Create the folder if it does not exist
    
    os.makedirs(args.cleaned_data, exist_ok=True)

    # Create the path
    path = os.path.join(args.cleaned_data, 'df_cleaned.csv')

    # Write the data preparation output as csv file
    df_cleaned.to_csv(path, index=False)

    run.complete()

if __name__ == '__main__':
    main()

