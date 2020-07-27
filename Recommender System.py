import findspark
findspark.init("D:\Spark")

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Ilk Ornek').getOrCreate()
lnes = spark.read.csv('ratings.csv',inferSchema=True,header=True)
lnes.show()
lnes.describe().show()
training,test = lnes.randomSplit([0.7,0.3])
als = ALS(maxIter=5,regParam=0.01,userCol='userId',itemCol='movieId',ratingCol='rating')
model = als.fit(training)
predictions = model.transform(test)
predictions.show()
single_user = test.filter(test['userId']==12).select(['movieId','userId'])
single_user.show()
rec = model.transform(single_user)
rec.orderBy('prediction',ascending=False).show()
evalate = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evalate.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))