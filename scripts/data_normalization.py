import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler

#---------------------------------------------------------------------------------------
# cReating argument parsers
#--------------------------------------------------------------------------------------

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, dest='cleaned_data', help='Input dataset')
parser.add_argument('--output', type=str, dest='normalized_data', help='Output path')

args = parser.parse_args()

#---------------------------------------------------------------------------------------
# Fetching data and doing normalization
#--------------------------------------------------------------------------------------

# Retrieve input data
df = pd.read_csv(os.path.join(args.cleaned_data, 'df_cleaned.csv'))

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_df = scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_df, columns=df.columns)

#------------------------------------------------------------------------------------


# Create output directory if it doesn't exist
os.makedirs(args.normalized_data, exist_ok=True)

# Save the normalized dataframe to the specified output path
output_path = os.path.join(args.normalized_data, 'normalized_df.csv')
normalized_df.to_csv(output_path, index=False)

#---------------------------------------------------------------------------------------
# Logging and submitting run
#--------------------------------------------------------------------------------------

# Get the run context
run = Run.get_context()

# Log the transformation process
run.log('Data normalization', 'Data is normalized using MinMaxScaler')

run.complete()
