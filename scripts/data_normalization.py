import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, dest='cleaned_data', help='Input dataset')
parser.add_argument('--output', type=str, dest='normalized_data', help='Output path')

args = parser.parse_args()

# Get the run context
run = Run.get_context()

# Retrieve input data
df = pd.read_csv(os.path.join(args.cleaned_data, 'df_cleaned.csv'))

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_df = scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_df, columns=df.columns)

# Saving output file locally
local_path = './outputs/df_normalized.csv'
os.makedirs('./outputs',exist_ok=True)
normalized_df.to_csv(local_path, index=False)


# Create output directory if it doesn't exist
os.makedirs(args.normalized_data, exist_ok=True)

# Save the normalized dataframe to the specified output path
output_path = os.path.join(args.normalized_data, 'normalized_df.csv')
normalized_df.to_csv(output_path, index=False)

# Log the transformation process
run.log('Data normalization', 'Data is normalized using MinMaxScaler')

run.complete()
