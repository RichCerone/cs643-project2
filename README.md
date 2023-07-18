# Project 2 Setup
This is the setup guide for project 2 setup and how to run the ML predication application.

## Setting Up the EMR Cluster
This project uses SageMaker notebooks running on a Spark cluster. I used this guide to setup the cluster: https://aws.amazon.com/blogs/machine-learning/build-amazon-sagemaker-notebooks-backed-by-spark-in-amazon-emr/

**The steps are broken out below:**

1. Open the AWS Management Console, and from Services menu at the top of the screen, select EMR under the Analytics section. Choose Create Cluster. Go to Advanced Options (at the top, next to where it says Create Cluster – Quick Options) and uncheck everything. Then, specifically check Livy and Spark. Choose Next.

2. Under Network, select Your VPC. You will also want to make a note of your EC2 Subnet because you will need this later.

3. Choose Next and then choose Create Cluster. Feel free to include any other options to your cluster that you think might be appropriate, such as adding key pairs for remote access to nodes, or a custom name to the cluster.

4. Now, you need to get your Private IP address for the Master node of your Spark cluster.

5. Choose Services and then choose EMR. Wait until your cluster is marked as Waiting (in green), and then choose the cluster you created. Choose the Hardware tab.

6. Choose your Master’s ID, and then scroll right to find Private IP Address. Save this for later.

### Set up Security Groups and Open Ports

1. Next we need to set up a security group and open the relevant ports, so our Amazon SageMaker notebook can talk to our Spark cluster via Livy on port 8998.

2. In the console, choose Services and then EC2. In the navigation pane at the left, choose Security Groups. Then choose the Create Security Group button.

3. Set a Security Group Name, a Description, and the VPC you used for your EMR cluster.

4. Choose Create. This creates the security group, making it possible for us to only open the port to instances that are in this group. We still need to open the port in our ElasticMapReduce-master group. While still in Security Groups, obtain the Group ID of your SageMaker notebook security group. Save this value for later.

5. We need to modify the EMR master security group that was automatically created when we created our EMR cluster. Select the ElasticMapReduce-master group, and then choose the Inbound tab. Choose the Edit button, and then choose the Add Rule button.

6. We want to create a Custom TCP Rule, on port 8998, and set the Security Group ID to the Group ID from the SageMaker notebook security group that we collected earlier.

7. Choose the Save button. You’ve now opened up the important ports, so your SageMaker notebook instance can talk to your EMR cluster over Livy.

### Set up SageMaker Notebook

1. We now have our EMR Spark cluster running with Livy, and the relevant ports available. Now let’s get our Amazon SageMaker Notebook instance up and running.

2. Choose Services and then Amazon SageMaker. Choose the Create Notebook Instance button.

3. You need to set a Notebook instance name and select a Notebook instance type. Be aware that there are some naming constraints for your Notebook instance name (maximum of 63 alphanumeric characters, can include hyphens but not spaces, and it must be unique within your account in an AWS Region). You also need to set up an IAM role with AmazonSageMakerFullAccess, plus access to any necessary Amazon Simple Storage Service (Amazon S3) buckets. You need to specify which buckets you want the role to have access to, but then you can let Amazon SageMaker generate the role for you.

4. Then you need to make sure that you set up your Notebook instance in the same VPC as your EMR cluster (for me, this was sagemaker-spark). Also select the same Subnet as the EMR cluster (you should have made note of this earlier, when you created your EMR cluster). Finally, set your security group to the group you created earlier for Notebook instances (mine was sagemaker-notebook).

5. Choose Create Notebook Instance.

6. Wait until EMR finishes provisioning the cluster and the SageMaker notebook status says InService.

### Connect the Notebook to EMR Cluster

1. Now we have our EMR Spark cluster and our Amazon SageMaker notebook running, but they can’t talk to each other yet. The next step is to set up Sparkmagic in SageMaker so it knows how to find our EMR cluster.

2. While still in the Amazon SageMaker console, go to your Notebook Instances and choose Open on the instance that was provisioned.

3. Inside your Jupyter console, choose New and then Terminal.

4. Type the following commands:
```
cd .sparkmagic
wget https://raw.githubusercontent.com/jupyter-incubator/sparkmagic/master/sparkmagic/example_config.json
mv example_config.json config.json
```

5. Then you need to edit the config.json, and replace every instance of `localhost` with the Private IP of your EMR Master that you used earlier. Use the following commands:

    - nano config.json
    - ctrl+\
    - localhost
    - {your EMR Master private IP}
    - a
    - ctrl+x
    - y
    - enter

   This should replace three instances of localhost in the “url” field of the three kernel credentials. Feel free to use any editor you are comfortable with, and save the changes.

## Training the ML Model
I used MLeap to train my model. The reason being that MLeap is a little faster than MLib.It also acts as a wrapper around MLib and provides the same functionality.

Below is my python code I used in my SageMaker notebook to train my model.

### Start and Configure Spark Instance
```python
sc
%%configure -f
{
    "conf": {
        "spark.jars.packages": "ml.combust.mleap:mleap-spark_2.12:0.20.0,ml.combust.mleap:mleap-spark-base_2.12:0.20.0",
        "spark.pyspark.python": "python3",
        "spark.pyspark.virtualenv.enabled": "true",
        "spark.pyspark.virtualenv.type": "native",
        "spark.pyspark.virtualenv.bin.path": "/usr/bin/virtualenv"
    }
}

sc.install_pypi_package("mleap==0.20.0")
sc.install_pypi_package("boto3")
```

### Create Schema for Data Input on Model
```python
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
```

### Get SageMaker Session and Set Bucket
We start a SageMaker session and then get the default bucket used by the SageMaker notebook.
```python
%%local
import sagemaker

session = sagemaker.Session()
bucket = sess.default_bucket()
```

Then I set the bucket as global variable in the Spark kernel:
```python
%%send_to_spark -i bucket -t str -n bucket
```

### Read Input Data and Set Split
We need to read in the CSV input to set what our model will train with. I also set a random split on the dataframes so that it uses 80% for training and 20% for validation. If you have not already, upload the input CSV to your desired bucket.

**Note: I had to conver the files from Canvas to comma delimitted CSV's.**

```python
dataframes = spark.read.csv(
    "s3://{your_bucket}/TrainingDataset.csv",
    header=True,
    schema=schema
)
dataframes.show(5) # Use this to ensure data was read properly.
(train_df, validation_df) = dataframes.randomSplit([0.8, 0.2])
```

### Define Feature Transformers
We need to transform the data into a single vector so we can feed it into our ML model which is using Random Forest.

```python
from pyspark.ml.feature VectorAssembler

assembler = VectorAssembler(
    inputCols=[
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality"
    ],
    outputCol="features",
)
```

### Define Random Forest Model and Train
```python
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(labelCol="quality", featuresCol="features", maxDepth=6, numTrees=18)
pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(train_df)
```

### Validate Transformed Data
I used this codeto validate the Model is able to predict data using the trained data.

```python
transformed_train_df = model.transform(train_df)
transformed_validation_df = model.transform(validation_df)
transformed_validation_df.select("prediction").show(5)
```

### Check Root Mean Square Error
This code checks to see if the RMSE is satisifactory. When I performed my validation I got **Train RMSE** of **0.146364** and **Validation RMSE** of **0.206627**. These are fairly low numbers and are close to 0 which is what we want to see in our model.

```python
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="quality", predictionCol="prediction", metricName="rmse")
train_rmse = evaluator.evaluate(transformed_train_df)
validation_rmse = evaluator.evaluate(transformed_validation_df)

print("Train RMSE = %g" % train_rmse)
print("Validation RMSE = %g" % validation_rmse)
```

### Serialze Model
We serialize the model so we can deploy it later as an endpoint service.

```python
model.serializeToBundle("jar:file:/tmp/model.zip", transformed_validation_df)
```

### Convert Model to tar.gz Format
This convert the model to a tar.gz for deployment purposes.

```python
import zipfile

with zipfile.ZipFile("/tmp/model.zip") as zf:
    zf.extractall("/tmp/model")

import tarfile

with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
    tar.add("/tmp/model/bundle.json", arcname="bundle.json")
    tar.add("/tmp/model/root", arcname="root")
```

### Upload Trained Model to S3:
```python
import os
import boto3

s3 = boto3.resource("s3")
file_name = os.path.join("emr/wine/mleap", "model.tar.gz")
s3.Bucket(bucket).upload_file("/tmp/model.tar.gz", file_name)
```



