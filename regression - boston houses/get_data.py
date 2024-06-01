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

p_df = df.pandas_api()
p_df.info()

# COMMAND ----------

p_df["ocean_proximity"].value_counts()

# COMMAND ----------

p_df.describe()

# COMMAND ----------

import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')


# COMMAND ----------



import matplotlib.pyplot as plt
cols = [
  # "longitude"         ,
  # "latitude"          ,
  "housing_median_age",
  # "total_rooms"       ,
  # "total_bedrooms"    ,
  # "population"        ,
  # "households"        ,
  # "median_income"     ,
  # "median_house_value",
]
p_df[cols].hist(bins=50, figsize=(20,15))


# COMMAND ----------

df.summary().display()

# COMMAND ----------

display(df)
