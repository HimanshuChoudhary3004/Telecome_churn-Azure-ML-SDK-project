import os
from azureml.core import Workspace, Dataset, Experiment, ComputeTarget, Run
from azureml.train.automl import AutoMLConfig
import pandas as pd
import joblib

# Connect to workspace
ws = Workspace.from_config('..azureml\config')
input_ds = Dataset.get_by_name(ws, 'Telecom_churn_dataset')
compute_target = ComputeTarget(workspace=ws, name='AML-CC-01')

# AutoML settings
auto_ml_settings = {        
                    "n_cross_validations": 3,
                    "primary_metric": 'norm_macro_recall',  # Use one of the supported metrics
                    "enable_early_stopping": True,
                    "experiment_timeout_hours": 1.0,
                    "max_concurrent_iterations": 1,
                    "max_cores_per_iteration": -1,
                    "log_metrics": ['accuracy', 'recall_score', 'precision_score', 'AUC_weighted', 'norm_macro_recall'] 
                    }


# AutoML configuration
automl_config = AutoMLConfig(
    task='classification',
    compute_target=compute_target,
    training_data=input_ds,
    label_column_name='Churn',
    featurization='auto',
    validation_size=0.3,
    **auto_ml_settings
)

# Submit AutoML experiment
automl_experiment = Experiment(workspace=ws, name='Automl_Experiment_Tele_01')
auto_ml_run = automl_experiment.submit(automl_config)
auto_ml_run.wait_for_completion(show_output=True)

#----------------------------------------------------------------------------------------------------
# Retrieve the latest run dynamically instead of hardcoding parent_run_id
#----------------------------------------------------------------------------------------------------

# Get the most recent completed run of the AutoML experiment
parent_run = auto_ml_run

# Retrieve all child runs under the parent run
all_runs = parent_run.get_children()

# Initialize an empty list to collect run metrics
automl_per_df = []

# Loop through all child runs and collect metrics
for run in all_runs:
    run_metrics = run.get_metrics()
    properties = run.get_properties()
    
    # Get the algorithm name from the run's properties
    algorithm_name = properties.get('run_algorithm', 'Unknown Algorithm')
    
    # Collect the run details (algorithm name, accuracy, recall, precision, AUC, norm recall)
    run_info = {
        'Algorithm': algorithm_name,
        'Run ID': run.id,
        'Accuracy': run_metrics.get('accuracy'),
        'Recall': run_metrics.get('recall_score'),
        'Precision': run_metrics.get('precision_score'),
        'AUC': run_metrics.get('AUC_weighted'),
        'Normalized Recall': run_metrics.get('norm_macro_recall')
    }
    
    automl_per_df.append(run_info)

# Convert the collected data into a pandas DataFrame for easy comparison
automl_per_df = pd.DataFrame(automl_per_df)

# Sort the DataFrame based on recall score (descending)
automl_per_df = automl_per_df.sort_values(by='Recall', ascending=False)

# Ensure the directory for saving the pickle file exists
output_dir = 'joblib'
os.makedirs(output_dir, exist_ok=True)

# Save the sorted DataFrame as a pickle file for further analysis
joblib.dump(value=automl_per_df, filename=os.path.join(output_dir, 'automl_per_df.pkl'))

# Display the sorted DataFrame
print(automl_per_df)
