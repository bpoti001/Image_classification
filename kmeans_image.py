from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
from pyspark import SparkContext
import sys
from numpy import sum
from scipy.spatial.distance import euclidean
import pickle
from numpy import array
import numpy as np

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))
sc=SparkContext(appName="stocks_clustring")
with open('/home/hadoop/success_vec_protocol_1.pickle','rb') as objectf:
    dictt = pickle.load(objectf)
keys_rdd = sc.parallelize(list(dictt.keys()))
key_value_rdd = keys_rdd.map(lambda x: (x,dictt[x].reshape((4096,))))
array_rdd = key_value_rdd.map(lambda (k,v):v)
result=[]
for i in range(2,100):
	clusters = KMeans.train(array_rdd, i, maxIterations=i*10, initializationMode="random")
	WSSSE = array_rdd.map(lambda point: error(point)).reduce(lambda x, y: x + y)
	print("Within Set Sum of Squared Error = " + str(WSSSE) + "for k = " +str(i))
	result.append("Within Set Sum of Squared Error = " + str(WSSSE) + "for k = " +str(i))
	clusters.save(sc, "/home/hadoop/cluster_model/cluster_model_k_"+str(i))

with open('result_list.pickle','a') as f:
	pickle.dump(result,f)

sc.close()
