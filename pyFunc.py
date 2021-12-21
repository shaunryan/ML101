# Databricks notebook source
import mlflow.pyfunc

# Define the model class
class AddN(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):
        return model_input.apply(lambda column: column + self.n)
      
add5_model = AddN(n=5)
model_path = "add_n_model"
model_name = "add_n_model"

# COMMAND ----------

# SAVE & LOAD

# Construct and save the model - saves in a relative path under the notebook id
mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_path)


# COMMAND ----------

# EVALUATE MODEL

import pandas as pd
model_input = pd.DataFrame([range(10)])
model_output = loaded_model.predict(model_input)
assert model_output.equals(pd.DataFrame([range(5, 15)]))
display(model_output)

# COMMAND ----------

# DELETE EXPERIMENT

# from mlflow.tracking import MlflowClient
# client = MlflowClient()
# client.delete_experiment("ce119dbb1e4146a6b37fa931e126a5d2")


# COMMAND ----------

# REGISTER MODEL
mlflow.pyfunc.log_model(
    python_model=add5_model,
    artifact_path=model_path,
    registered_model_name=model_name
)

# COMMAND ----------

# DELETE RUN

import mlflow

run_id = "ec2fbdac823041d7941b5d2886724244"
# with mlflow.start_run() as run:
#     mlflow.log_param("p", 0)

run = mlflow.get_run(run_id)

run_id = run.info.run_id
mlflow.delete_run(run_id)

print("run_id: {}; lifecycle_stage: {}".format(run_id, mlflow.get_run(run_id).info.lifecycle_stage))

# COMMAND ----------

# DELETE REGISTERED MODEL

from mlflow.tracking import MlflowClient

model_name = "add_n_model"
client = MlflowClient()
versions=[1, 2, 3]
for version in versions:
    try:
      client.delete_model_version(name=model_name, version=version)
    except:
      print(f"model {model_name} version {version} not found")

# Delete a registered model along with all its versions
try:
  client.delete_registered_model(name=model_name)
except:
  print(f"{model_name} not found")

