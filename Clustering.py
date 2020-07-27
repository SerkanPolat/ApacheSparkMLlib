import findspark
findspark.init("D:\Spark")

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Clustering').getOrCreate()
data = spark.read.csv('seeds_dataset.csv',inferSchema=True,header=True)
data.printSchema()
data.columns

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['area','perimeter','compactness','length_of_kernel','width_of_kernel','asymmetry_coefficient','length_of_groove'])
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler

vecAsemb = VectorAssembler(inputCols=data.columns,outputCol='features')
final_data = vecAsemb.transform(data)
scalar = StandardScaler(inputCol="features",outputCol='scaledFeatures',withStd=True,withMean=True)
scalarModel = scalar.fit(final_data)
final_data = scalarModel.transform(final_data)

kmeans = KMeans(featuresCol='scaledFeatures').setK(3).setSeed(1)
model = kmeans.fit(final_data)
wsse = model.computeCost(final_data)
wsse
model.transform(final_data).select('prediction').show()
