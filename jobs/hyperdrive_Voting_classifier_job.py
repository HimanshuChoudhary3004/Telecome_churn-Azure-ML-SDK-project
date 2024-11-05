from azureml.core import Workspace, Dataset, Environment, Experiment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.train.hyperdrive import HyperDriveConfig, BayesianParameterSampling, choice, uniform, PrimaryMetricGoal
from azureml.core import ScriptRunConfig
import os
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------
# Initialize workspace
ws = Workspace.from_config('.azureml/config')
# Load dataset
input_ds = Dataset.get_by_name(workspace=ws, name='Telecom_churn_dataset').as_named_input('input_data')
# Load environment
my_env = Environment.get(workspace=ws, name='Telecome_churn env', version='6')

#----------------------------------------------------------------------------------------------------------------
# Define compute target
cluster_name = 'AML-CC-02'
if cluster_name in ws.compute_targets:
    print('Compute Target exists. Accessing it...')
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
else:
    print('Creating and firing up compute cluster...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size='Standard_E16s_v3', max_nodes=2)
    compute_target = ComputeTarget.create(workspace=ws, name=cluster_name, provisioning_configuration=provisioning_config)

compute_target.wait_for_completion(show_output=True)

#-------------------------------------------------------------------------------------------------------------------------------

# Script run configuration
script_run_config = ScriptRunConfig(source_directory='.', 
                                     script='voting_classifier_training_script.py', 
                                     arguments=['--input', input_ds],
                                     environment=my_env,
                                     compute_target=compute_target)
        


# Define hyperparameters for HyperDrive
hyper_params = BayesianParameterSampling({
    "--n_estimators": choice([100, 500]),
    "--max_depth": choice([10, 50]),
    "--min_samples_split": choice([2, 10]),
    "--min_samples_leaf": choice([1, 10]),
    "--max_features": choice(['sqrt', 'log2']),
    "--criterion": choice(['gini', 'entropy']),
    "--max_samples": uniform(0.3, 1.0),

    # Hyperparameters for Gradient Boosting
    "--learning_rate": uniform(0.01, 0.3),
    "--subsample": uniform(0.5, 1.0),
    "--max_depth_gb": choice([3, 5, 7]),
    "--min_samples_split_gb": choice([2, 5, 10]),
    "--min_samples_leaf_gb": choice([1, 2, 4]),
})



# HyperDrive configuration
hd_config = HyperDriveConfig(
    hyperparameter_sampling=hyper_params,
    primary_metric_name='recall',
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_concurrent_runs=3,
    max_total_runs=260,
    max_duration_minutes=500,
    policy=None,
    run_config=script_run_config
)

#-------------------------------------------------------------------------------------------------------------------------------

# Experiment setup
experiment = Experiment(ws, name='Hyperdrive_voting_classifier_Telecome_01')
print("Submitting HyperDrive job...")
run = experiment.submit(hd_config)

# Wait for completion
run.wait_for_completion(show_output=True)
#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
# Get best run details
best_run = run.get_best_run_by_primary_metric()

best_run_metrics = best_run.get_metrics()
best_run_hyper_param = best_run.get_details()['runDefinition']['arguments']

# Visualization print statements
print(f'\nBest run ID is: {best_run.id}')

print('\nBest Hyper Parameters are:')
for i in range(0, len(best_run_hyper_param), 2):
    print(f'{best_run_hyper_param[i]}: {best_run_hyper_param[i + 1]}')

print('\nMetrics of best run are:')
for metric_name, metric_value in best_run_metrics.items():
    print(f'{metric_name}: {metric_value}')

#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
# Create the 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Prepare data for visualization
metric_names = list(best_run_metrics.keys())
metric_values = list(best_run_metrics.values())

# Create a bar chart for metrics
plt.figure(figsize=(10, 5))
plt.bar(metric_names, metric_values, color='blue')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Metrics of Best Run')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot to the 'plots' directory
plt.savefig('plots/best_run_metrics01.png')
#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
# Register the model from the best run
print("Registering the model from the best run...")

# Define hyperparameters dictionary
hyperparameters = {best_run_hyper_param[i]: best_run_hyper_param[i + 1] for i in range(0, len(best_run_hyper_param), 2)}

model = best_run.register_model(model_path = 'outputs/voting_classifier_model.pkl',
                                model_name = 'Telecome_voting_classifier_model',
                                tags = {'Source' : 'Hyperdriver best run','Algorithm' : 'Voting_classifier'},
                                properties = {
                                    'accuracy': best_run_metrics.get('accuracy'),
                                    'Recall': best_run_metrics.get('recall'),
                                    'Precision': best_run_metrics.get('precision'),
                                    'f1_score': best_run_metrics.get('f1'),
                                    'AUC_Score': best_run_metrics.get('auc'),
                                    'Best Hyperparameters': str(hyperparameters)},
                                description = 'Voting classifier with Random forest and gradient boosting algoritm are used')

print("Model registered successfully with name:", model.name)
#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
# Delete the compute cluster
compute_target.delete()
