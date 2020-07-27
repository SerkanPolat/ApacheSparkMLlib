import findspark
findspark.init('D:/Spark')
from pyspark import SparkContext
sc = SparkContext()
liste1 = [('Fizik',73),('Matematik',60),('Kimya',66),('Matematik',25),('Fizik',10)]
rdd1 = sc.parallelize(liste1)
rdd1.reduceByKey(lambda x,y:x+y).collect()
rdd1.groupByKey().collect()
liste2 = [('Fizik',273),('Matematik',460),('Kimya',366),('Matematik',725),('Fizik',410)]
rdd2 = sc.parallelize(liste2+liste1)
toplam = rdd2.combineByKey(lambda x:x(x,1),(lambda x,y:(x[0]+y,x[1]+1)),(lambda x,y:(x[0]+y[0],x[1]+y[1])))
toplam.collect()
rdd1.join(rdd2).collect()
rdd1.rightOuterJoin(rdd2).collect()
rdd1.leftOuterJoin(rdd2).collect()
rdd1.cogroup(rdd2).collect()
rdd1.countByKey()
rdd2.countByValue()
rdd2.lookup('Fizik')
