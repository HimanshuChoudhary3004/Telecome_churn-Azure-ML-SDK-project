from azureml.core import Workspace, Dataset, Environment, Datastore, ScriptRunConfig, Experiment, RunConfiguration, ComputeTarget
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal, choice, uniform, GridParameterSampling, RandomParameterSampling, BayesianParameterSampling
from azureml.core.compute import AmlCompute, ComputeTarget

ws = Workspace.from_config('.azureml\config')
dataset = Dataset.get_by_name(workspace=ws, name='Telecom_churn_dataset')
input_data = dataset.as_named_input('row_data')
my_env = Environment.get(workspace=ws, name='Telecome_churn env', version='6')

cluster_name = 'AML-CC-01'

if cluster_name in ws.compute_targets:
    print('Compute target exist , acessing it ...')
    computetarget = ComputeTarget(workspace=ws, name=cluster_name)

else:
    print('Creating compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size='Standard_E16s_v3',
                                                                max_nodes=1)
    computetarget = ComputeTarget.create(workspace=ws,
                                        name=cluster_name,
                                        provisioning_configuration=provisioning_config)




script_run_config = ScriptRunConfig(source_directory = '.',
                                    script = 'RFC_training_script.py',
                                    arguments = ['--input',input_data],
                                    environment = my_env,
                                    compute_target = computetarget,
                                   )



hyper_params = BayesianParameterSampling(
                                    {
                                        "--n_estimators": choice([100, 500]),
                                        "--max_depth": choice([10, 50]),
                                        "--min_samples_split": choice([2, 10]),
                                        "--min_samples_leaf": choice([1, 10]),
                                        "--max_features": choice(['sqrt', 'log2']),
                                        "--criterion": choice(['gini', 'entropy']),
                                        "--max_samples": uniform(0.3, 1.0),                                   
                                    }                                  
                                     )


hd_config = HyperDriveConfig(
                             hyperparameter_sampling=hyper_params,
                             primary_metric_name='recall',
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             policy=None,
                             run_config=script_run_config,
                             max_total_runs=162,
                             max_concurrent_runs=1,
                             max_duration_minutes=120
                             )



experiment = Experiment(workspace=ws, name = 'Hyperdrive_RFC_Telecome_004')
hyperdrive_run = experiment.submit(config=hd_config)
hyperdrive_run.wait_for_completion(show_output=True)


best_run = hyperdrive_run.get_best_run_by_primary_metric()



import pandas as pd
from azureml.core import Run

# Get all runs and their metrics for the HyperDrive experiment
run_details = []
for run in hyperdrive_run.get_children():
    run_metrics = run.get_metrics()
    run_params = run.get_details()['runDefinition']['arguments']
    
    # Extract hyperparameters from arguments
    hyperparameters = {}
    for i in range(0, len(run_params), 2):  # Arguments are passed as key-value pairs
        hyperparameters[run_params[i]] = run_params[i + 1]

    # Merge hyperparameters and metrics into a single dictionary
    run_data = {**hyperparameters, **run_metrics}
    
    # Add the run ID to the dictionary
    run_data['Run ID'] = run.id
    
    # Append this run's data to the list
    run_details.append(run_data)

# Convert the run details into a DataFrame
results_df = pd.DataFrame(run_details)

# Display the DataFrame
print(results_df)

# Save the DataFrame to a CSV file
results_df.to_csv('hyperdrive_run_results.csv', index=False)
print("Results saved to hyperdrive_run_results.csv")





# Get hyperparameters and metrics of the best run
best_run_params = best_run.get_details()['runDefinition']['arguments']
best_run_metrics = best_run.get_metrics()

# Print the best run ID
print(f"\nBest Run ID: {best_run.id}")

# Define hyperparameters dictionary
print("\nBest Run Hyperparameters:")
hyperparameters = {best_run_hyper_param[i]: best_run_hyper_param[i + 1] for i in range(0, len(best_run_hyper_param), 2)}
print(hyperparameters)
# Print best run's metrics
print("\nBest Run Metrics:")
for metric_name, metric_value in best_run_metrics.items():
    print(f"{metric_name}: {metric_value}")



# Register the model from the best run
print("Registering the model from the best run...")
model = best_run.register_model(model_path='outputs/random_forest_model.pkl',
                                model_name='Telecome_churn_RFC',
                                tags={'source': 'hyperdrive-best-run', 'algorithm': 'Random_Forest_Classifier'},
                                properties={
                                    'accuracy': best_run_metrics.get('accuracy'),
                                    'Recall': best_run_metrics.get('recall'),
                                    'Precision': best_run_metrics.get('precision'),
                                    'f1_score': best_run_metrics.get('f1'),
                                    'AUC_Score': best_run_metrics.get('auc'),
                                    'Best Hyperparameters': str(hyperparameters)
                                },
                                description='Random Forest Classifier model trained on the Telecom churn dataset using hyperdrive optimization.')

print("Model registered successfully with name:", model.name)
