import findspark
findspark.init("D:\Spark")

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('myproj').getOrCreate()
data = spark.read.csv('titanic.csv',inferSchema=True,header=True)
data.printSchema()
data.columns
my_cols = data.select(['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
my_final_data = my_cols.na.drop()
from pyspark.ml.feature import (VectorAssembler,OneHotEncoder,StringIndexer)
gender_indexer = StringIndexer(inputCol='Sex',outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol='SexIndex',outputCol='SexVec')
embark_indexer = StringIndexer(inputCol='Embarked',outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex',outputCol='EmbarkVec')
assembler = VectorAssembler(inputCols=['Pclass','SexVec','Age','SibSp','Parch','Fare','EmbarkVec'],outputCol='features')
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
log_reg_titanic = LogisticRegression(featuresCol='features',labelCol='Survived')
pipeline = Pipeline(stages=[gender_indexer,embark_indexer,gender_encoder,embark_encoder,assembler,log_reg_titanic])  
train_titanic_data, test_titanic_data = my_final_data.randomSplit([0.6,0.4])
fit_model = pipeline.fit(train_titanic_data)
results = fit_model.transform(test_titanic_data)
my_eval = RegressionEvaluator(labelCol='Survived')
my_eval.setPredictionCol("prediction")
results.select('Survived','prediction').show()
AUC = my_eval.evaluate(results)
AUC