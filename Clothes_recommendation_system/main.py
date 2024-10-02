import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName('movieProjectt').config("spark.memory.offHeap.enabled", "true").config("spark.memory.offHeap.size", "10g").getOrCreate()

# Read Data
spark_df = spark.read.csv('data.csv', header=True, inferSchema=True)
print(spark_df.show())
print(spark_df.count())
print(spark_df.select("street").distinct().count())
print(spark_df.groupBy('city').agg(countDistinct("street").alias("City_Count")).show())
print(spark_df.groupBy('city').agg(countDistinct("bedrooms").alias("Bedrooms_Count")).orderBy(desc('Bedrooms_Count')).show())
print(spark_df.select(min('date')).show())

df = spark_df.withColumn("from_date", lit("12/01/10 2:00"))
df = df.withColumn("from_date", to_timestamp("from_date", 'yy/MM/dd H:mm'))
print(df.select(col("date").cast("long")).show())
df2 = df.withColumn("from_date", to_timestamp(col("from_date"))).withColumn("recency", col("date").cast("long") - col("from_date").cast("long"))
print(df2.show())

print(spark_df.printSchema())
print(spark_df.columns)
print(spark_df.select(['bathrooms', 'floors']).show())
print(spark_df.dtypes)
print(spark_df.describe().show())

spark_df = spark_df.withColumn('sqft', spark_df['sqft_lot'] - spark_df['sqft_above'] - spark_df['sqft_basement'])
spark_df = spark_df.drop('sqft')
spark_df = spark_df.withColumnRenamed('sqft_lot', 'Squared_footage_lot')
spark_df.na.drop().show()
print(spark_df.show())

spark_df.na.fill('Missing').show()



### sql connection

spark2 = SparkSession.builder.appName('NewProject').config("spark.memory.offHeap.enabled", "true").config("spark.memory.offHeap.size", "10g").getOrCreate()

df = spark2.read.csv('data2.csv', header=True, inferSchema=True)
print(df.show())

spark2.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df = df.withColumn('date',to_timestamp("InvoiceDate", 'dd/MM/yyyy HH:mm'))
print(df.show())
df.select(max("date")).show()

print(df.show())

df = df.withColumn("from_date", lit("08/1/10 08:26"))
print(df.show())
df = df.withColumn('from_date',to_timestamp("from_date", 'yy/MM/dd HH:mm'))

df2=df.withColumn('from_date',to_timestamp(col('from_date'))).withColumn('recency',col("date").cast("long") - col('from_date').cast("long"))
print(df2.show())

df2 = df2.join(df2.groupBy('CustomerID').agg(max('recency').alias('recency')),on='recency',how='leftsemi')
print(df2.show())

df_freq = df2.groupBy('InvoiceNo').agg(count('InvoiceDate').alias('frequency'))
print(df_freq.show())

df3 = df2.join(df_freq, on='InvoiceNo', how='inner')
print(df3.show())

m_val = df3.withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))
print(m_val.show())

total_amount_customer = m_val.groupBy("InvoiceNo").agg(sum('TotalAmount').alias("SumTotalAmount"))
print(total_amount_customer.show())

final_df = m_val.join(df3, on="InvoiceNo", how="inner")
final_df = final_df.drop('recency')
print(final_df.show())

finaldf = final_df.select(['recency', 'frequency', 'SumTotalAmount', 'InvoiceNo']).distinct()
print(finaldf.show())


## Machine Learning

spark_build = SparkSession.builder.appName('machinelearning').config("spark.memory.offHeap.enabled", "true").config("spark.memory.offHeap.size", "10g").getOrCreate()

df = spark_build.read.csv('data.csv', header=True, inferSchema=True)
print(df.show())

df = df.drop('date', 'street', 'city', 'statezip', 'country', '_c18','_c19','_c20','_c21','_c22','_c23','_c24')
print(df.show())
print(df.printSchema())

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

assemble = VectorAssembler(inputCols=[
    'price', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'
], outputCol='features')

assembled_data = assemble.transform(df)
print(assembled_data.show())

scale = StandardScaler(inputCol='features', outputCol='standarized')
data_scale = scale.fit(assembled_data)
data_scale_output = data_scale.transform(assembled_data)
print(data_scale_output.show())

data_scale_output.select('standarized').show(2, truncate=False)

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np

cost = np.zeros(10)

evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standarized', metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2, 10):
    KMeans_algo = KMeans(featuresCol='standarized', k=i)
    KMeans_fit = KMeans_algo.fit(data_scale_output)
    output = KMeans_fit.transform(data_scale_output)
    cost[i] = KMeans_fit.summary.trainingCost


import pandas as pd
import pylab as pl

print(cost[2:])

df_cost = pd.DataFrame(cost[2:])
df_cost.columns = ["cost"]
new_col = range(2, 10)
df_cost.insert(0, 'cluster', new_col)
pl.plot(df_cost.cluster, df_cost.cost)
pl.xlabel("Number of Clusters")
pl.ylabel("Score")
pl.title("Elbow Curve")
pl.show()


KMeans_algo = KMeans(featuresCol='standarized', k=4)
KMeans_fit = KMeans_algo.fit(data_scale_output)

preds = KMeans_fit.transform(data_scale_output)
preds.show(5, 0)

import matplotlib.pyplot as plt
import seaborn as sns

df_viz = preds.select('price', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'prediction')
df_viz = df_viz.toPandas()
avg_df = df_viz.groupby(['prediction'], as_index=False).mean()

list1 = ['price', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

for i in list1:
    sns.barplot(x='prediction',y=str(i),data=avg_df)
    plt.show()




###
