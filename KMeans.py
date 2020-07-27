import findspark
findspark.init('D:\Spark')

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Hack_Yakalama').getOrCreate()
data = spark.read.csv('hack_data.csv',inferSchema=True,header=True)
data.printSchema()
data.columns

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler

assembler = VectorAssembler(inputCols=['Session_Connection_Time','Bytes Transferred','Kali_Trace_Used','Servers_Corrupted','Pages_Corrupted','WPM_Typing_Speed'],outputCol='features')
DataFinal = assembler.transform(data)
scalar = StandardScaler(inputCol='features',outputCol='scaledFeatures',withStd=True,withMean=True)
scalarModel = scalar.fit(DataFinal)
DataFinal = scalarModel.transform(DataFinal)
for k in range(2,9):
    kmeans = KMeans(featuresCol='scaledFeatures').setK(k)
    model = kmeans.fit(DataFinal)
    wsse = model.computeCost(DataFinal)
    print("With K={}".format(k))
    print("Within Set Sum of Squared Errors = " + str(wsse))
    print('--'*30)
    
    
    
    
