import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer, JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from dotenv import load_dotenv
import boto3
import pandas as pd
import os

load_dotenv()

boto3_session = boto3.session.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name=os.getenv("REGION_NAME")
)
sagemaker_client = boto3.client("sagemaker", os.getenv("REGION_NAME"))
sagemaker_runtime_client = boto3.client("sagemaker-runtime", os.getenv("REGION_NAME"))

session = sagemaker.Session(
    boto_session=boto3_session,
    sagemaker_client=sagemaker_client,
    sagemaker_runtime_client=sagemaker_runtime_client,
)

endpoint_name = os.getenv("ENDPOINT_NAME")

with open(os.getenv("INPUT_FILE"), 'r') as f:
    data = pd.read_csv(os.getenv("INPUT_FILE"))

for i in data.index:
    payload = f"{data['fixed acidity'][i]},{data['volatile acidity'][i]},{data['citric acid'][i]},{data['residual sugar'][i]},{data['chlorides'][i]},{data['free sulfur dioxide'][i]},{data['total sulfur dioxide'][i]},{data['density'][i]},{data['pH'][i]},{data['sulphates'][i]},{data['alcohol'][i]},{data['quality'][i]}"
    predictor = Predictor(
            endpoint_name=endpoint_name, sagemaker_session=session, serializer=CSVSerializer()
        )
    r = predictor.predict(payload)
    print(r)