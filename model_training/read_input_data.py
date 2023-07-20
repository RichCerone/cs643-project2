dataframes = spark.read.csv(
    "s3://{your_bucket}/TrainingDataset.csv",
    header=True,
    schema=schema
)
dataframes.show(5) # Use this to ensure data was read properly.
(train_df, validation_df) = dataframes.randomSplit([0.8, 0.2])