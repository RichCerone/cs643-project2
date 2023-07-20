import json
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sparkml.model import SparkMLModel

schema = {
    "input": [
        {"name": "fixed_acidity", "type": "double"},
        {"name": "volatile_acidity", "type": "double"},
        {"name": "citric_acid", "type": "double"},
        {"name": "residual_sugar", "type": "double"},
        {"name": "chlorides", "type": "double"},
        {"name": "free_sulfur_dioxide", "type": "double"},
        {"name": "total_sulfur_dioxide", "type": "double"},
        {"name": "density", "type": "double"},
        {"name": "pH", "type": "double"},
        {"name": "sulphates", "type": "double"},
        {"name": "alcohol", "type": "double"},
        {"name": "quality", "type": "double"},
    ],
    "output": {"name": "prediction", "type": "double"},
}
schema_json = json.dumps(schema, indent=2)

boto3_session = boto3.session.Session()
sagemaker_client = boto3.client("sagemaker")
sagemaker_runtime_client = boto3.client("sagemaker-runtime")

# Initialize sagemaker session
session = sagemaker.Session(
    boto_session=boto3_session,
    sagemaker_client=sagemaker_client,
    sagemaker_runtime_client=sagemaker_runtime_client,
)

role = get_execution_role()

sparkml_data = "s3://{}/{}/{}".format(bucket, "emr/wine/mleap", "model.tar.gz")
model_name = "sparkml-wine"
sparkml_model = SparkMLModel(
    model_data=sparkml_data,
    role=role,
    spark_version="3.3",
    sagemaker_session=session,
    name=model_name,
    env={"SAGEMAKER_SPARKML_SCHEMA": schema_json},
)

endpoint_name = "sparkml-wine-ep"
sparkml_model.deploy(
    initial_instance_count=1, instance_type="ml.c4.xlarge", endpoint_name=endpoint_name
)

# Test

from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer, JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import pandas as pd

s3 = boto3.client('s3')

with open('ValidationDataset.csv', 'wb') as f:
    s3.download_fileobj('cs-643-ml-bucket', 'ValidationDataset.csv', f)

    data = pd.read_csv('ValidationDataset.csv')
    
for i in data.index:
    payload = f"{data['fixed acidity'][i]},{data['volatile acidity'][i]},{data['citric acid'][i]},{data['residual sugar'][i]},{data['chlorides'][i]},{data['free sulfur dioxide'][i]},{data['total sulfur dioxide'][i]},{data['density'][i]},{data['pH'][i]},{data['sulphates'][i]},{data['alcohol'][i]},{data['quality'][i]}"
    predictor = Predictor(
            endpoint_name=endpoint_name, sagemaker_session=session, serializer=CSVSerializer()
        )
    r = predictor.predict(payload)
    print(r)