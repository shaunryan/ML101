# Databricks notebook source
import os
import tarfile
import urllib
import pandas as pd


# COMMAND ----------

NAME = "housing"
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("temp", "datasets", NAME)
HOUSING_URL = DOWNLOAD_ROOT + f"datasets/{NAME}/{NAME}.tgz"


def fetch_housing_data(
  dataset_name:str=NAME, 
  housing_url=HOUSING_URL, 
  housing_path=HOUSING_PATH
):
  os.makedirs(housing_path, exist_ok=True)
  tgz_path = os.path.join(housing_path, f"{dataset_name}.tgz")
  urllib.request.urlretrieve(housing_url, tgz_path)
  housing_tgz = tarfile.open(tgz_path)
  housing_tgz.extractall(path=housing_path)
  housing_tgz.close()


def load_housing_data(path:str):
  csv_path = os.path.join(path, "housing.csv")
  print(f"loading {csv_path}")
  return pd.read_csv(csv_path)


  

# COMMAND ----------


fetch_housing_data()
df:pd.DataFrame = load_housing_data(HOUSING_PATH)

df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Random Splitting
# MAGIC </br>
# MAGIC
# MAGIC - will create a different test set each time it's executed!
# MAGIC - persisting the test set will solve this
# MAGIC - or using a seed
# MAGIC - If there data is updated then the test_set will lose reproducibility

# COMMAND ----------

import numpy as np

def split_train_test(data: pd.DataFrame, test_ratio:float = 0.2):
  # seeding np will create a consistent test_set on the same data
  np.random.seed(42)
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  # get from the start to 20%
  test_indices = shuffled_indices[:test_set_size]
  # get from 20% to the end
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(df)

# COMMAND ----------

train_set.describe()

# COMMAND ----------

test_set.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Indentity Splitting
# MAGIC </br>
# MAGIC
# MAGIC - Requires a consistent ID
# MAGIC - Is consistent following updates

# COMMAND ----------

from zlib import crc32


def test_set_check(
  identifier:int, 
  test_ratio:float = 0.2
  ):
  """ Compute a hash of the id and put it in the test
      set if the hash is <= 20% of the maximum hash value
  """
  return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(
  data:pd.DataFrame, 
  id_column:str, 
  test_ratio:float = 0.2
):
  """ New test will contain 20% of the instances
      but will not contain any instance that was previously in the training set
  """
  ids = data[id_column]
  in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
  return data.loc[~in_test_set], data.loc[in_test_set]


# the housing data does not have an indentifier column
# so we will use the row number
df_with_id = df.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(df_with_id, "index")



# COMMAND ----------

train_set.describe()

# COMMAND ----------

test_set.describe()
