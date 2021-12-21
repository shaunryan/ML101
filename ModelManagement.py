# Databricks notebook source
# MAGIC %md # Register a new model
# MAGIC There are three ways to register a model, i.e. create a model version:
# MAGIC   1. `MlflowClient.create_model_version`
# MAGIC   2. `<flavor>.log_model`
# MAGIC   3. `mlflow.register_model`

# COMMAND ----------

import uuid
prefix = uuid.uuid4().hex[0:4]  # A unique prefix for model names to avoid clashing
model1_name = f'{prefix}_model1'
model2_name = f'{prefix}_model2'
model3_name = f'{prefix}_model3'

# COMMAND ----------

import mlflow
import mlflow.pyfunc

class SampleModel(mlflow.pyfunc.PythonModel):
  def predict(self, ctx, input_df):
      return 7

artifact_path = 'sample_model'

# COMMAND ----------

# MAGIC %md ##### Create a model version via `MlflowClient.create_model_version`

# COMMAND ----------

# Log a model to MLflow Tracking
from mlflow.tracking.artifact_utils import get_artifact_uri

with mlflow.start_run() as new_run:
  mlflow.pyfunc.log_model(  
      python_model=SampleModel(),
      artifact_path=artifact_path,
  )
  run1_id = new_run.info.run_id
  source = get_artifact_uri(run_id=run1_id, artifact_path=artifact_path)

print(source)

# COMMAND ----------

# Instantiate an MlflowClient pointing to the local tracking server and a remote registry server
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

model = client.create_registered_model(model1_name)
client.create_model_version(name=model1_name, source=source, run_id=run1_id)

# COMMAND ----------

# MAGIC %md At this point, you should be able to see the new model version in the model registry workspace.

# COMMAND ----------

# MAGIC %md ##### Create a model version via `mlflow.register_model`
# MAGIC The method will also create the registered model if it does not already exist.

# COMMAND ----------

mlflow.register_model(model_uri=f'runs:/{run1_id}/{artifact_path}', name=model2_name)

# COMMAND ----------

# MAGIC %md ##### Create a model version via `<flavor>.log_model`
# MAGIC The method will also create the registered model if it does not already exist.

# COMMAND ----------


with mlflow.start_run() as new_run:
  mlflow.pyfunc.log_model(
    python_model=SampleModel(),
    artifact_path=artifact_path,
    registered_model_name=model3_name, # specifying this parameter automatically creates the model & version
  )

# COMMAND ----------

# MAGIC %md
# MAGIC # Getting Model Versions

# COMMAND ----------

def getModelVersions(model_name:str):
  
  client = MlflowClient()
  model_versions = client.search_model_versions(f"name='{model_name}'")

#   for mv in model_versions:
#       pprint(dict(mv), indent=4)

  versions = [dict(mv)["version"] for mv in model_versions]

  return versions

vs = getModelVersions(model3_name)

for v in vs:
  print(v)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Promote Model

# COMMAND ----------

# stage = "None"
stage = "Staging"
# stage = "Production"
# stage = "Archive"
version = 1

client = MlflowClient()
client.transition_model_version_stage(
    name = model3_name,
    version = version,
    stage = stage
)

# COMMAND ----------

# MAGIC %md # Download model from registry

# COMMAND ----------

# MAGIC %md ###### Method 1: `mlflow.<flavor>.load_model` with explicitly-specified registry URI

# COMMAND ----------

model = mlflow.pyfunc.load_model(f'models:/{model3_name}/1')
model.predict(1)

# COMMAND ----------

# MAGIC %md ###### Method 3: Use `ModelsArtifactRepository`
# MAGIC If you want to download the model files without loading into a model framework, you can use an `ArtifactRepository`. `ModelsArtifactRepository` is the most convenient subclass for model registry operations. To specify a remote registry, you can either set `registry_uri` via `mlflow.set_registry_uri`, or pass in the registry information directly into `ModelsArtifactRepository` as below.

# COMMAND ----------

import os

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
local_path = ModelsArtifactRepository(f'models:/{model3_name}/1').download_artifacts('')
os.listdir(local_path)

# COMMAND ----------

# MAGIC %md ###### Method 4: Get the location of the model files and download via the REST API (`DbfsRestArtifactRepository`)
# MAGIC In Python, Databricks recommends using `ModelsArtifactRepository.download_artifacts` (Method 3), which is equivalent to this method. However, this example is useful for understanding how you can perform the download using the REST API in other contexts.

# COMMAND ----------

import os
import shutil
from six.moves import urllib
from mlflow.store.artifact.dbfs_artifact_repo import DbfsRestArtifactRepository

version = client.get_latest_versions(model3_name, ['None'])[0].version
uri = client.get_model_version_download_uri(model3_name, version)
path = urllib.parse.urlparse(uri).path
local_path = DbfsRestArtifactRepository(f'dbfs:{path}').download_artifacts('')
print(f"""
  version: {version}
  uri: {uri}
  path: {path}
  local path: {local_path}
""")

print(os.listdir(local_path))
shutil.rmtree(local_path)


# COMMAND ----------

# MAGIC %md
# MAGIC # Managing Model Experiments & Runs

# COMMAND ----------

# MAGIC %md ###### Runs: Get all the runs for an experiment

# COMMAND ----------

import mlflow


def getExperimentRuns(name:str=None, id:int=-1, run_view_type:int=1, max_results:int=1000, order_by=None):
  
  try:
    if name:
      exp = mlflow.get_experiment_by_name(name)
    elif id > 0:
      exp = mlflow.get_experiment(id)
    else:
      raise Exception(f"experiment name or id must be provided name={name} id={id}")
      
  except:
    raise Exception(f"experiment can't be found name={name} id={id}")
    
  experiment_id = dict(exp)["experiment_id"]
  runs = mlflow.list_run_infos(experiment_id, run_view_type, max_results, order_by)
  
  return runs

runs = getExperimentRuns(name="/ML101/RegisterModel")
runs = getExperimentRuns(id=2631785103320342)

display(runs)

# COMMAND ----------

def getExperimentId(name:str):
  
  try:
    if name:
      exp = mlflow.get_experiment_by_name(name)
    else:
      raise Exception(f"experiment name must be provided name={name}")
      
  except:
    raise Exception(f"experiment can't be found name={name} id={id}")
    
  experiment_id = dict(exp)["experiment_id"]
  
  return experiment_id

# COMMAND ----------

# MAGIC %md # Clean up
# MAGIC Delete the models, as well as the intermediate copies of model artifacts.

# COMMAND ----------

def delete_version_tmp_files(version, registry_uri):
  import posixpath
  location = posixpath.dirname(version.source)
  print(f"Deleting: {location}")
  if registry_uri == 'databricks':
    dbutils.fs.rm(location, recurse=True)
  else:
    from mlflow.utils.databricks_utils import get_databricks_host_creds
    from mlflow.utils.rest_utils import http_request
    import json
    response = http_request(
      host_creds=get_databricks_host_creds(registry_uri), 
      endpoint='/api/2.0/dbfs/delete',
      method='POST',
      json=json.loads('{"path": "%s", "recursive": "true"}' % (location))
    )

def archive_and_delete(name):
  try:
    client.transition_model_version_stage(name, 1, 'Archived')
  finally:
    client.delete_registered_model(name)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Delete Runs

# COMMAND ----------

runs = getExperimentRuns(id=2631785103320342)

def delete_run(run_id:str):
  mlflow.delete_run(r._run_id)
  print(f"run_id: {run_id}; lifecycle_stage: {mlflow.get_run(run_id).info.lifecycle_stage}")

for r in runs:
  delete_run(r._run_id)


# COMMAND ----------

# MAGIC %md
# MAGIC ###### Delete Experiment

# COMMAND ----------

from mlflow.tracking import MlflowClient

experiment_name = "/ML101/RegisterModel"

experiment_id=getExperimentId(experiment_name)

client = MlflowClient()
client.delete_experiment(experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###### Delete Model Versions

# COMMAND ----------

from mlflow.tracking import MlflowClient

versions = getModelVersions(model3_name)

def delete_model_versions(model_name:str, versions:list=None):
  
  if not versions:
    _versions = versions
  else:
    _versions = getModelVersions(model_name)
  
  client = MlflowClient()
  for version in _versions:
      try:
        client.delete_model_version(name=model_name, version=version)
      except:
        print(f"model {model_name} version {version} not found")

delete_model_versions(model3_name, [1, 2])
versions = getModelVersions(model3_name)
delete_model_versions(model3_name, versions)
delete_model_versions(model3_name)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###### Deleting Models

# COMMAND ----------

# delete_version_tmp_files(client.get_model_version(model1_name, 1), "databricks")
# delete_version_tmp_files(client.get_model_version(model2_name, 1), "databricks")
# delete_version_tmp_files(client.get_model_version(model3_name, 1), "databricks")

archive_and_delete(model1_name)
archive_and_delete(model2_name)
archive_and_delete(model3_name)
