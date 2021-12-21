# Databricks notebook source
# MAGIC %md #### Set-up
# MAGIC 
# MAGIC We will log models to the MLflow tracking server in _this_ workspace, and register the models into the model registry in another workspace. 
# MAGIC 
# MAGIC 1. In the model registry workspace, create an access token.
# MAGIC 2. In this workspace, create secrets and store the access token and the remote workspace information. The easiest way is to use the Databricks CLI, but you can also use the Secrets REST API.
# MAGIC   1. Create a secret scope: `databricks secrets create-scope --scope <scope>`.
# MAGIC   2. Pick a unique name for the target workspace, which we'll refer to as `<prefix>`. Then create three secrets:
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-host`. Enter the hostname of the model registry workspace.
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-token`. Enter the access token from the model registry workspace.
# MAGIC     * `databricks secrets put --scope <scope> --key <prefix>-workspace-id`. Enter the workspace ID for the model registry workspace which can be found in the URL of any page in the workspace.
# MAGIC 
# MAGIC **Before running the notebook, enter the secret scope and key prefix corresponding to the remote registry workspace in the notebook parameter fields above.**

# COMMAND ----------

dbutils.widgets.text('1_registry_secret_scope', '')
dbutils.widgets.text('2_registry_secret_key_prefix', '')
scope = str(dbutils.widgets.get('1_registry_secret_scope'))
key = str(dbutils.widgets.get('2_registry_secret_key_prefix'))

registry_uri = 'databricks://' + scope + ':' + key if scope and key else None

# COMMAND ----------

# MAGIC %md #### Register a new model
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

# COMMAND ----------

# Instantiate an MlflowClient pointing to the local tracking server and a remote registry server
from mlflow.tracking.client import MlflowClient
client = MlflowClient(tracking_uri=None, registry_uri=registry_uri)

model = client.create_registered_model(model1_name)
client.create_model_version(name=model1_name, source=source, run_id=run1_id)

# COMMAND ----------

# MAGIC %md At this point, you should be able to see the new model version in the model registry workspace.

# COMMAND ----------

# MAGIC %md ##### Create a model version via `mlflow.register_model`
# MAGIC The method will also create the registered model if it does not already exist.

# COMMAND ----------

mlflow.set_registry_uri(registry_uri)
mlflow.register_model(model_uri=f'runs:/{run1_id}/{artifact_path}', name=model2_name)

# COMMAND ----------

# MAGIC %md ##### Create a model version via `<flavor>.log_model`
# MAGIC The method will also create the registered model if it does not already exist.

# COMMAND ----------

mlflow.set_registry_uri(registry_uri)

with mlflow.start_run() as new_run:
  mlflow.pyfunc.log_model(
    python_model=SampleModel(),
    artifact_path=artifact_path,
    registered_model_name=model3_name, # specifying this parameter automatically creates the model & version
  )

# COMMAND ----------

# MAGIC %md #### Managing models on a remote model registry

# COMMAND ----------

client = MlflowClient(tracking_uri=None, registry_uri=registry_uri)

# COMMAND ----------

model_names = [m.name for m in client.list_registered_models() if m.name.startswith(prefix)]
print(model_names)

# COMMAND ----------

client.update_registered_model(model2_name, description='For ranking')
client.get_registered_model(model2_name)

# COMMAND ----------

client.transition_model_version_stage(model3_name, 1, 'Staging')
client.get_model_version(model3_name, 1)

# COMMAND ----------

model1_version = client.get_model_version(model1_name, 1)  # for cleaning up files later
client.delete_registered_model(model1_name)
assert model1_name not in [m.name for m in client.list_registered_models()]

# COMMAND ----------

# MAGIC %md #### Download model from remote registry

# COMMAND ----------

# MAGIC %md ###### Method 1: `mlflow.<flavor>.load_model` with explicitly-specified registry URI

# COMMAND ----------

model = mlflow.pyfunc.load_model(f'models://{scope}:{key}@databricks/{model3_name}/Staging')
model.predict(1)

# COMMAND ----------

# MAGIC %md ###### Method 2: `mlflow.<flavor>.load_model` with `registry_uri` set

# COMMAND ----------

mlflow.set_registry_uri(registry_uri)
model = mlflow.pyfunc.load_model(f'models:/{model3_name}/Staging')
model.predict(1)

# COMMAND ----------

# MAGIC %md ###### Method 3: Use `ModelsArtifactRepository`
# MAGIC If you want to download the model files without loading into a model framework, you can use an `ArtifactRepository`. `ModelsArtifactRepository` is the most convenient subclass for model registry operations. To specify a remote registry, you can either set `registry_uri` via `mlflow.set_registry_uri`, or pass in the registry information directly into `ModelsArtifactRepository` as below.

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
local_path = ModelsArtifactRepository(f'models://{scope}:{prefix}@databricks/{model3_name}/Staging').download_artifacts('')
os.listdir(local_path)

# COMMAND ----------

# MAGIC %md ###### Method 4: Get the location of the model files and download via the REST API (`DbfsRestArtifactRepository`)
# MAGIC In Python, Databricks recommends using `ModelsArtifactRepository.download_artifacts` (Method 3), which is equivalent to this method. However, this example is useful for understanding how you can perform the download using the REST API in other contexts.

# COMMAND ----------

import os
from six.moves import urllib
from mlflow.store.artifact.dbfs_artifact_repo import DbfsRestArtifactRepository

version = client.get_latest_versions(model3_name, ['Staging'])[0].version
uri = client.get_model_version_download_uri(model3_name, version)
path = urllib.parse.urlparse(uri).path
local_path = DbfsRestArtifactRepository(f'dbfs://{scope}:{key}@databricks{path}').download_artifacts('')
os.listdir(local_path)

# COMMAND ----------

# MAGIC %md #### Clean up
# MAGIC Delete the models, as well as the intermediate copies of model artifacts.

# COMMAND ----------

def delete_version_tmp_files(version):
  import posixpath
  location = posixpath.dirname(version.source)
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

delete_version_tmp_files(model1_version)
delete_version_tmp_files(client.get_model_version(model2_name, 1))
delete_version_tmp_files(client.get_model_version(model3_name, 1))

archive_and_delete(model2_name)
archive_and_delete(model3_name)
