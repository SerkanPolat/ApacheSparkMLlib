import findspark
findspark.init("D:\Spark")

from pyspark import SparkContext
sc = SparkContext()
liste1 = [1,2,2,4,4,6,7,8,9]
rdd1 = sc.parallelize(liste1)
rdd1.map(lambda x:x+1).collect()
rdd1.filter(lambda x:x%2).collect()
rdd1.flatMap(lambda x:(x,x*100,999)).collect()
rdd1.map(lambda x:(x,x*100,99)).collect()
rdd1.distinct().collect()
liste2 = [100,101,2,3]
rdd2 = sc.parallelize(liste2)
rdd1.union(rdd2).collect()
rdd1.subtract(rdd2).collect()
rdd2.subtract(rdd1).collect()
liste3 = ['serkan','polat','rdd','denemeleri']
rdd3 = sc.parallelize(liste3)
rdd1.cartesian(rdd3).count()
rdd4 = sc.parallelize([1,2,3,4])
rdd4.reduce(lambda x,y:x+y)
rdd4.fold(3,lambda x,y:x*y)
rdd4.take(4)
rdd4.top(3)
