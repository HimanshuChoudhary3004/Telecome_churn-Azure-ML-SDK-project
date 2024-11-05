from azureml.core import Workspace, Dataset, Experiment, Run
import numpy as np 
import pandas as pd 
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
    df_cleaned = pd.get_dummies(df,drop_first=True)
    
    feat_importance = abs(df_cleaned.corr()['Churn']).sort_values(ascending=False)[1:]


 
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

