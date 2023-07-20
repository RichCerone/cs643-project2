from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(labelCol="quality", featuresCol="features", maxDepth=6, numTrees=18)
pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(train_df)