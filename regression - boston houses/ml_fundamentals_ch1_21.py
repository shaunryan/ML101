# Databricks notebook source
# todo https://learn.microsoft.com/en-us/azure/databricks/_static/notebooks/mlflow/mlflow-end-to-end-example.html

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# better life index
root = "./datasets/lifesat/"
oecd_bli_path = f"{root}/oecd_bli_2015.csv"
gdp_per_capita = f"{root}/gdp_per_capita.csv"


print(f"""
  oecd_bli_path = {oecd_bli_path}
  gdp_per_capita = {gdp_per_capita}
""")



# COMMAND ----------

# load the data

oecd_bli = pd.read_csv(oecd_bli_path, thousands=',')
gdp_per_capita = pd.read_csv(gdp_per_capita, thousands=",", delimiter="\t", encoding="latin1", na_values="n/a")

# COMMAND ----------

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# COMMAND ----------

# prepare the data

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# COMMAND ----------

# visualise the data
country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Linear Regression
# MAGIC
# MAGIC Model based learning

# COMMAND ----------

import sklearn.linear_model

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# train the model
model.fit(x, y)

# COMMAND ----------

# make a prediction about Cyprus

x_new = [[22587]] # Cyprus's GDP per capita
print(model.predict(x_new))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # K-Nearest Neighbors
# MAGIC
# MAGIC Instance based learning. Takes the nearest k data points and averages them.

# COMMAND ----------

import sklearn.neighbors

# select k-nearest neighbors model

model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

model.fit(x, y)

# COMMAND ----------

# make a prediction about Cyprus

x_new = [[22587]] # Cyprus's GDP per capita
print(model.predict(x_new))
