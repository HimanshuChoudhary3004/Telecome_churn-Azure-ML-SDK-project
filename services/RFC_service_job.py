from azureml.core import Workspace, Environment, ComputeTarget
from azureml.core.compute import AksCompute
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AksWebservice
from azureml.exceptions import ComputeTargetException, WebserviceException

# Load the workspace
ws = Workspace.from_config('.azureml/config')

# Get the environment
my_env = Environment.get(workspace=ws, name='Telecome_churn env', version='6')

#--------------------------------------------------------------------------------------------
# Production cluster
cluster_name = 'Aks-cluster-002'


if cluster_name in ws.compute_targets:
    print('Cluster exists, accessing it...')
    production_cluster = ComputeTarget(workspace=ws, name=cluster_name)
else:
    print('Creating and firing up compute target...')
    provisioning_config = AksCompute.provisioning_configuration(agent_count=1,
                                                                vm_size='STANDARD_D11_V2',
                                                                location='centralindia',
                                                                cluster_purpose='DevTest')
    production_cluster = ComputeTarget.create(workspace=ws, name=cluster_name, provisioning_configuration=provisioning_config)
        
    production_cluster.wait_for_completion(show_output=True)


#--------------------------------------------------------------------------------------------------------------------------------------------

# Creating Inference configuration
inference_config = InferenceConfig(entry_script='Telecome_churn_scoring_script.py',
                                   source_directory='.\\services',
                                   environment=my_env)

#------------------------------------------------------------------------------------------------------------------------------------

# Creating deployment configuration
deploy_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

#--------------------------------------------------------------------------------------------------------------------------------------

# Deploy webservice
model = ws.models['Telecom_voting_classifier']

try:
    service = model.deploy(workspace=ws,
                           name='telecome-churn-prediction',
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=deploy_config,
                           deployment_target=production_cluster)

    service.wait_for_deployment(show_output=True)

except WebserviceException as e:
    print(f"Webservice Exception: {e}")
    # Handle specific web service deployment errors here
    if "insufficient compute resource" in str(e):
        print("Deployment failed due to insufficient compute resources. Consider adjusting your configuration.")
    else:
        print("An unknown error occurred during service deployment.")
    exit(1)  # Exit the script if deployment fails
