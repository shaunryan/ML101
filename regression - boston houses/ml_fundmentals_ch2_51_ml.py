# Databricks notebook source
import os
import tarfile
import urllib


# COMMAND ----------

NAME = "housing"
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("temp", "datasets", NAME)
HOUSING_URL = DOWNLOAD_ROOT + f"datasets/{NAME}/{NAME}.tgz"


def fetch_housing_data(dataset_name:str=NAME, housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
  os.makedirs(housing_path, exist_ok=True)
  tgz_path = os.path.join(housing_path, f"{dataset_name}.tgz")
  urllib.request.urlretrieve(housing_url, tgz_path)
  housing_tgz = tarfile.open(tgz_path)
  housing_tgz.extractall(path=housing_path)
  housing_tgz.close()

fetch_housing_data()
  

# COMMAND ----------

from pyspark.sql.types import *

def load_housing_data(dataset_name:str=NAME, housing_path=HOUSING_PATH):

    schema = StructType([
      StructField("longitude", DoubleType(), False),
      StructField("latitude", DoubleType(), False),
      StructField("housing_median_age", DoubleType(), False),
      StructField("total_rooms", DoubleType(), False),
      StructField("total_bedrooms", DoubleType(), False),
      StructField("population", DoubleType(), False),
      StructField("households", DoubleType(), False),
      StructField("median_income", DoubleType(), False),
      StructField("median_house_value", DoubleType(), False),
      StructField("ocean_proximity", StringType(), False)
    ])
    csv_path = os.path.join(housing_path, f"{dataset_name}.csv")
    csv_path = os.path.abspath(csv_path)
    csv_path = f"file:{csv_path}"
    print(f"loading data from {csv_path}")
    options = {
      "header": True
    }
    df = (spark.read
      .format("csv")
      .schema(schema)
      .options(**options)
      .load(csv_path))
    return df

df = load_housing_data()
df.display()

# COMMAND ----------

housing = df.pandas_api()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Test Set

# COMMAND ----------

# DBTITLE 1,Pandas
import numpy as np

def split_train_test(data, test_ratio):

  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  training_data, testing_data = data.iloc[train_indices], data.iloc[test_indices]

  return training_data, testing_data

train_set, test_set = split_train_test(housing, 0.2)



# COMMAND ----------

print(f"""
total = {df.count()}
train_set = {len(train_set)}
test_set = {len(test_set)}

train_set_% = {len(train_set) / df.count()}
test_set_% = {len(test_set) / df.count()}
""")



# COMMAND ----------

# DBTITLE 1,Pyspark
train_set, test_set = df.randomSplit([0.8, 0.2])

print(f"""
total = {df.count()}
train_set = {train_set.count()}
test_set = {test_set.count()}

train_set_% = {train_set.count() / df.count()}
test_set_% = {test_set.count() / df.count()}
""")

# COMMAND ----------


