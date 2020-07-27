import findspark
findspark.init("D:\Spark")

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('IlkUygulama').getOrCreate()
dataset = spark.read.format('csv').option('header','true').load('veri.csv')
data = dataset.groupBy('Platform').count().orderBy('count',ascesding=False)
data.collect()
data2 = dataset.groupBy('Genre').count().orderBy('count',ascending=False)
data2.collect()


