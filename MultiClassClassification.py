import findspark
findspark.init('D:\Spark')
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('MultiLayer').getOrCreate()
data = spark.read.csv('iris.data',inferSchema=True,header=False)
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['_c0','_c1','_c2','_c3'],outputCol='features')
final_data = assembler.transform(data)
splits = final_data.randomSplit([0.6,0.4])
train = splits[0]
test = splits[1]
layers = [4,5,4,3]
trainer = MultilayerPerceptronClassifier(maxIter=100,layers=layers,labelCol='_c4')
model = trainer.fit(final_data)
result = model.transform(test)
predictionAndLabels = result.select("prediction", "_c4")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",labelCol='_c4')
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
