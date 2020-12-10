#!/usr/bin/env python:
# coding: utf-8

# In[87]:


import findspark
findspark.init('/home/ubuntu/sparkfolder/spark-2.4.7-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('wine_quality_predictor').getOrCreate()
sc = spark.sparkContext


# In[88]:


# load contents of TrainingDataset.csvto spark DataFrma
# we need to specify the custom separator `;`
wine = spark.read.csv('/home/ubuntu/Project/TrainingDataset.csv',header='true', inferSchema='true')


# In[89]:



# let's see the schema and the number of rows
wine.printSchema()


# In[90]:


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


# In[91]:


wine = wine.drop(wine.columns[0])


# In[92]:


wine.show()


# In[93]:


wine.columns


# In[94]:


from pyspark.sql import functions as F

for col in wine.columns:
  wine = wine.withColumn(
    col,
    F.col(col).cast("double")
  )

wine.show()


# In[95]:


#change datatype of quality to integer
from pyspark.sql.types import IntegerType, DoubleType
wine = wine.withColumn('quality', wine['quality'].cast(IntegerType()))


# In[96]:


wine.printSchema()


# In[97]:


print("Rows: %s" % wine.count())
print(wine.limit(20))


# In[98]:


from pyspark.ml.feature import VectorAssembler

# select the columns to be used as the features (all except `quality`)
featureColumns = [c for c in wine.columns if c != 'quality']

# create and configure the assembler
assembler = VectorAssembler(inputCols=featureColumns, 
                            outputCol="features")

# transform the original data
dataDF = assembler.transform(wine)
dataDF.printSchema()


# In[99]:


print(dataDF.limit(3))


# In[100]:


from pyspark.ml.regression import LinearRegression

# fit a `LinearRegression` model using features in colum `features` and label in column `quality`
lr = LinearRegression(maxIter=30, regParam=0.3, elasticNetParam=0.3, featuresCol="features", labelCol="quality")
lrModel = lr.fit(dataDF)


# In[101]:


for t in zip(featureColumns, lrModel.coefficients):
    print t


# In[102]:


# predict the quality, the predicted quality will be saved in `prediction` column
predictionsDF = lrModel.transform(dataDF)
print(predictionsDF.limit(3))


# In[103]:


from pyspark.ml.evaluation import RegressionEvaluator

# create a regression evaluator with RMSE metrics

evaluator = RegressionEvaluator(
    labelCol='quality', predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictionsDF)
print("Root Mean Squared Error (RMSE) = %g" % rmse)


# In[104]:


from pyspark.sql.functions import *

# calculate the average wine quality
avgQuality = wine.groupBy().avg('quality').first()[0]
print(avgQuality)

# compute the 'zero' model predictions
# `lit` function creates a 'literal' column that is column with the provided value in all rows
zeroModelPredictionsDF = dataDF.select(col('quality'), lit(avgQuality).alias('prediction'))

# evaluate the 'zero' model
zeroModelRmse = evaluator.evaluate(zeroModelPredictionsDF)
print("RMSE of 'zero model' = %g" % zeroModelRmse)


# In[105]:


# split the input data into traning and test dataframes with 70% to 30% weights
(trainingDF, testDF) = wine.randomSplit([0.7, 0.3])


# In[106]:


from pyspark.ml import Pipeline

# construct the `Pipeline` that with two stages: the `vector assembler` and `regresion model estimator`
pipeline = Pipeline(stages=[assembler, lr])

# train the pipleline on the traning data
lrPipelineModel = pipeline.fit(trainingDF)

# make predictions
traningPredictionsDF = lrPipelineModel.transform(trainingDF)
testPredictionsDF = lrPipelineModel.transform(testDF)

# evaluate the model on test and traning data
print("RMSE on traning data = %g" % evaluator.evaluate(traningPredictionsDF))

print("RMSE on test data = %g" % evaluator.evaluate(testPredictionsDF))


# In[107]:


from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

# create a search grid with the cross-product of the parameter values (9 pairs)
search_grid = ParamGridBuilder()     .addGrid(lr.regParam, [0.0, 0.3, 0.6])     .addGrid(lr.elasticNetParam, [0.4, 0.6, 0.8]).build()

# use `CrossValidator` to tune the model
cv = CrossValidator(estimator = pipeline, estimatorParamMaps = search_grid, evaluator = evaluator, numFolds = 3)
cvModel = cv.fit(trainingDF)


# In[108]:


# evaluate the tuned model
cvTestPredictionsDF = cvModel.transform(testDF)
print("RMSE on test data with CV = %g" % evaluator.evaluate(cvTestPredictionsDF))


# In[109]:



from pyspark.ml.regression import RandomForestRegressor

# define the random forest estimator
rf = RandomForestRegressor(featuresCol="features", labelCol="quality", numTrees=100, maxBins=128, maxDepth=20,minInstancesPerNode=5, seed=33)
rfPipeline = Pipeline(stages=[assembler, rf])

# train the random forest model
rfPipelineModel = rfPipeline.fit(trainingDF)


# In[110]:


rfTrainingPredictions = rfPipelineModel.transform(trainingDF)
rfTestPredictions = rfPipelineModel.transform(testDF)
print("Random Forest RMSE on traning data = %g" % evaluator.evaluate(rfTrainingPredictions))
print("Random Forest RMSE on test data = %g" % evaluator.evaluate(rfTestPredictions))


# In[112]:



rfModel = rfPipelineModel.stages[1]
rfModel.featureImportances


# In[62]:
gateway = sc._gateway
java_list = gateway.jvm.java.util.ArrayList()
java_list.add(rfModel._java_obj)
modelRdd = sc._jsc.parallelize(java_list)
modelRdd.saveAsObjectFile("hdfs://ec2-3-88-182-126.compute-1.amazonaws.com:9000/home/ubuntu/sparkfolder/spark-2.4.7-bin-hadoop2.7/output/rf.model") 

