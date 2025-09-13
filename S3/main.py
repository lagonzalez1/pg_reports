import boto3
from botocore.exceptions import BotoCoreError, ClientError

## Asuuming the base role for CLI
s3 = boto3.client('s3')


class S3Instance:

    def __init__(self, bucket):
        self.bucket = bucket
    
    def put_object(self, key, body)-> bool:
        try:
            s3.put_object(
                Bucket=self.bucket,
                Key=str("student_reports/"+key),
                Body=body,
                ContentType='application/json'
            )
            return True
        except (BotoCoreError, ClientError) as e:
            return False
    


    