import findspark
findspark.init('D:/Spark')
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
spark = spark = SparkSession.builder.appName('LineerRegresyon').getOrCreate()
veri = spark.read.csv('Ecommerce Customers.csv',inferSchema=True,header=True)
veri.printSchema()
veri.show()
veri.head()
veri.show()
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['Avg Session Length','Time on App','Time on Website','Length of Membership'],outputCol='features')
VeriVec = assembler.setHandleInvalid("skip").transform(veri)
VeriVec.show()
VeriVec.printSchema()


SonVeri = VeriVec.select('features','Yearly Amount Spent')
egitimVeri,testVeri = SonVeri.randomSplit([0.6,0.4])
egitimVeri.show()

lr = LinearRegression(labelCol='Yearly Amount Spent')
lrModel = lr.fit(egitimVeri)
print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))
sonuclar = lrModel.evaluate(testVeri)
sonuclar.residuals.show()
print("RMSE: {}".format(sonuclar.rootMeanSquaredError))
print("MSE: {}".format(sonuclar.meanSquaredError))
