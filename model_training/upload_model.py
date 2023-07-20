import os
import boto3

s3 = boto3.resource("s3")
file_name = os.path.join("emr/wine/mleap", "model.tar.gz")
s3.Bucket(bucket).upload_file("/tmp/model.tar.gz", file_name)