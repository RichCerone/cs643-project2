from mleap.pyspark.spark_support import SimpleSparkSerializer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.types import StructField, StructType, StringType, DoubleType

schema = StructType(
    [
        StructField("fixed_acidity", DoubleType(), True),
        StructField("volatile_acidity", DoubleType(), True),
        StructField("citric_acid", DoubleType(), True),
        StructField("residual_sugar", DoubleType(), True),
        StructField("chlorides", DoubleType(), True),
        StructField("free_sulfur_dioxide", DoubleType(), True),
        StructField("total_sulfur_dioxide", DoubleType(), True),
        StructField("density", DoubleType(), True),
        StructField("pH", DoubleType(), True),
        StructField("sulphates", DoubleType(), True),
        StructField("alcohol", DoubleType(), True),
        StructField("quality", DoubleType(), True)
    ]
)