#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import findspark
findspark.init('/home/ubuntu/sparkfolder/spark-2.4.7-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('wine_quality_predictor').getOrCreate()
sc =spark.sparkContext

# In[ ]:


# load contents of TrainingDataset.csvto spark DataFrma
# we need to specify the custom separator `;`
wine = spark.read.csv('/home/ubuntu/Project/ValidationDataset.csv',header='true', inferSchema='true')


# In[ ]:


# let's see the schema and the number of rows
wine.printSchema()


# In[ ]:


from pyspark.sql import functions

split_col = pyspark.sql.functions.split(wine['"""""fixed acidity"""";""""volatile acidity"""";""""citric acid"""";""""residual sugar"""";""""chlorides"""";""""free sulfur dioxide"""";""""total sulfur dioxide"""";""""density"""";""""pH"""";""""sulphates"""";""""alcohol"""";""""quality"""""'], ';')
wine = wine.withColumn('fixed acidity', split_col.getItem(0))
wine = wine.withColumn('volatile acidity', split_col.getItem(1))
wine = wine.withColumn('citric acid', split_col.getItem(2))
wine = wine.withColumn('residual sugar', split_col.getItem(3))
wine = wine.withColumn('chlorides', split_col.getItem(4))
wine = wine.withColumn('free sulfur dioxide', split_col.getItem(5))
wine = wine.withColumn('total sulfur dioxide', split_col.getItem(6))
wine = wine.withColumn('density', split_col.getItem(7))
wine = wine.withColumn('pH', split_col.getItem(8))
wine = wine.withColumn('sulphates', split_col.getItem(9))
wine = wine.withColumn('alcohol', split_col.getItem(10))
wine = wine.withColumn('quality', split_col.getItem(11))
sorted(wine.columns)


# In[ ]:


wine = wine.drop(wine.columns[0])


# In[ ]:


wine.show()


# In[ ]:


wine.columns


# In[ ]:


from pyspark.sql import functions as F

for col in wine.columns:
  wine = wine.withColumn(
    col,
    F.col(col).cast("double")
  )

wine.show()


# In[ ]:


#change datatype of quality to integer
from pyspark.sql.types import IntegerType, DoubleType
wine = wine.withColumn('quality', wine['quality'].cast(IntegerType()))


# In[ ]:


wine.printSchema()


# In[ ]:


print("Rows: %s" % wine.count())
print(wine.limit(20))


# In[ ]:
from pyspark.ml.feature import VectorAssembler

# select the columns to be used as the features (all except `quality`)
featureColumns = [c for c in wine.columns if c != 'quality']

# create and configure the assembler
assembler = VectorAssembler(inputCols=featureColumns,
                            outputCol="features")

# transform the original data
dataDF = assembler.transform(wine)
dataDF.printSchema()


# calculate the average wine quality
avgQuality = wine.groupBy().avg('quality').first()[0]
print(avgQuality)


from pyspark.ml.classification import RandomForestClassificationModel

rfObjectFileLoaded = sc._jsc.objectFile("hdfs://ec2-3-88-182-126.compute-1.amazonaws.com:9000/home/ubuntu/sparkfolder/spark-2.4.7-bin-hadoop2.7/output/rf.model")
rfModelLoaded_JavaObject = rfObjectFileLoaded.first()
rfModelLoaded = RandomForestClassificationModel(rfModelLoaded_JavaObject)
loadedPredictionsDF = rfModelLoaded.transform(wine)


# evaluate the model again to see if we get the same performance
print("Loaded model RMSE = %g" % evaluator.evaluate(loadedPredictionsDF))


