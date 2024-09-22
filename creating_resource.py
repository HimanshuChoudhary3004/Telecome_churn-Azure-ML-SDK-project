import azureml.core
from azureml.core import Workspace, Datastore, Dataset

#---------------------------------------------------------------------------------------------------
# Creating workspce and writing config file in root directory
#----------------------------------------------------------------------------------------------------
ws = Workspace.create(name = 'HIM-ML-WS-01',
                     subscription_id='411d5123-f744-4298-8f59-203dad1675d2',
                     resource_group='AML-RG-01',
                     create_resource_group=True,
                     location='centralindia')

# saving config file to root directory
ws.write_config(path='.')



#-------------------------------------------------------------------------------------------
# Creating/Registering Datastore for 'Telecome customer churn' dataset at storage account = 'mldata01'
#--------------------------------------------------------------------------------------------
account_key = '----------------------------------------------------------------'

Datastore_conn =  Datastore.register_azure_blob_container(workspace=ws,
                                                          datastore_name='telecome_datastore_conn',
                                                          container_name='ml-blob-01',
                                                          account_name='mldatatest01',
                                                          account_key=account_key)




#--------------------------------------------------------------------------------------------------------
# Creating and registering Telecome dataset
#--------------------------------------------------------------------------------------------------------

# path of dataset at azure storage account
csv_path = [(Datastore_conn,'customer churn data\Telco-Customer-Churn.csv')]

# creating tabular dataset from csv path
Tele_dataset = Dataset.Tabular.from_delimited_files(path=csv_path)

# Registering dataset in workspace
Tele_dataset = Tele_dataset.register(workspace=ws,
                                     name='Telecom_churn_dataset',
                                     create_new_version=True)




from azureml.core import Environment, compute

#-------------------------------------------------------------------------------------------------------
#   Creating compute cluster for pipeline
#-------------------------------------------------------------------------------------------------------

#Importing compute libraries
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException

# Definning compute configuration
compute_name = 'AML-CC-01'
provisioning_Compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_DS11_v2',
                                                                    min_nodes=0,
                                                                    max_nodes=4)
# Creating compute instance 
compute_target = ComputeTarget.create(workspace=ws,
                                     name=compute_name,
                                     provisioning_configuration=provisioning_Compute_config)





# -----------------------------------------------------------------------------------------------------------
# Creating virtual environment for pipeline
# ------------------------------------------------------------------------------------------------------------

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies


myenv = Environment('Telecomee_churn_env_1')

# Define the dependencies (conda + pip)
myenv_dep = CondaDependencies.create(
    conda_packages=['scikit-learn', 'pandas', 'numpy'],  # Specify conda packages if any
    pip_packages=['azureml-sdk']  # Include any additional pip packages like azureml-sdk if needed
)

# Attach the dependencies to the environment
myenv.python.conda_dependencies = myenv_dep

# Optionally, enabling Docker if using it for the environment
myenv.docker.enabled = True
myenv.docker.base_image = 'mcr.microsoft.com/azureml/base:latest'  # Specify a base image if needed

myenv.register(workspace=ws)





