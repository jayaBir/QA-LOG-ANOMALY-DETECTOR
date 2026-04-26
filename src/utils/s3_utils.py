import boto3
import os

s3 = boto3.client("s3")

def download_file(bucket, key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

def upload_file(local_path, bucket, key):
    s3.upload_file(local_path, bucket, key)