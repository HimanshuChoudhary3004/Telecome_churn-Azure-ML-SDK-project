from azureml.core import Workspace, Dataset, Datastore, Experiment, Environment, ComputeTarget
from azureml.core.runconfig import RunConfiguration

# Accessing\Loading workspace, Enviornment and compute cluster

ws= Workspace.from_config('.azureml\config')
input_data = Dataset.get_by_name(ws,'Telecom_churn_dataset')
my_env = Environment.get(ws,'Telecomee_churn_env')
run_config = RunConfiguration()
run_config.environment = my_env
compute_target = ComputeTarget(ws,'AML-CC-01')


# Creating pipeline steps 
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep


cleaned_data = PipelineData('cleaned_data', ws.get_default_datastore())

Data_cleaning = PythonScriptStep(name="Data_cleaning_Step",
                                source_directory='.\scripts',               
                                script_name='data_cleaning.py',
                                inputs=[input_data.as_named_input('row_data')],
                                outputs=[cleaned_data],
                                arguments=['--input','row_data','--output', cleaned_data],
                                compute_target=compute_target,
                                runconfig=run_config
                                )


normalized_data = PipelineData('normalized_data', ws.get_default_datastore())

Data_normalization = PythonScriptStep(name='Data_normalization',
                                      source_directory='.\scripts',
                                      script_name='data_normalization.py',
                                      inputs=[cleaned_data],
                                      outputs=[normalized_data],
                                      arguments=['--input', cleaned_data,'--output',normalized_data],
                                      compute_target=compute_target,
                                      runconfig=run_config,
                                      )

split_data = PipelineData('split_data',ws.get_default_datastore())
Data_splitting = PythonScriptStep(name='Data_splitting',
                                  source_directory='.\scripts',
                                  script_name='data_split.py',
                                  inputs=[normalized_data],
                                  outputs=[split_data],
                                  arguments=['--input',normalized_data,'--output',split_data],
                                  compute_target=compute_target,
                                  runconfig=run_config)


model_output_data = PipelineData('model_output_data',ws.get_default_datastore())
Train_model = PythonScriptStep(name= 'Train_model',
                               source_directory='.\scripts',
                               script_name='train_model.py',
                               inputs=[split_data],
                               outputs=[model_output_data],
                               arguments=['--input',split_data,'--output',model_output_data],
                               compute_target=compute_target,
                               runconfig=run_config)




steps = [Data_cleaning,Data_normalization,Data_splitting,Train_model]

tel_pipeline = Pipeline(ws,steps)
experiment = Experiment(workspace=ws,name='Telecom_churn_pipeline_experiment')

new_run = experiment.submit(config=tel_pipeline)

new_run.wait_for_completion()