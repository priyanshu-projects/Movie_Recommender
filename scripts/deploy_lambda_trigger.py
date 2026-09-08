"""
scripts/deploy_lambda_trigger.py

Deploys an AWS Lambda function that triggers GitHub Actions retraining workflow
whenever new user interaction data is uploaded to S3.
"""

import json
import logging
import time
from pathlib import Path
import boto3

logger = logging.getLogger(__name__)

REGION = "ap-south-1"
BUCKET = "movie-recommender-mlops-745600"
LAMBDA_NAME = "movie-recommender-s3-trigger"
ROLE_NAME = "movie-recommender-lambda-role"


LAMBDA_CODE = """
import os
import json
import urllib.request

GITHUB_REPO = os.environ.get("GITHUB_REPO", "priyanshu-projects/Movie_Recommender")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

def lambda_handler(event, context):
    print("S3 Event received:", json.dumps(event))
    
    # Trigger GitHub Actions workflow dispatch
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/retrain.yml/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
        "User-Agent": "AWS-Lambda-Trigger"
    }
    data = json.dumps({"ref": "main"}).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            print("✓ GitHub Actions workflow triggered successfully! Response:", resp.status)
            return {"statusCode": resp.status, "body": "Workflow triggered"}
    except Exception as exc:
        print("❌ Failed to trigger GitHub Actions:", str(exc))
        return {"statusCode": 500, "body": str(exc)}
"""


def deploy():
    session = boto3.Session(region_name=REGION)
    iam = session.client("iam")
    lambda_cli = session.client("lambda")
    s3 = session.client("s3")

    # 1. Create IAM Role for Lambda
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for S3 event trigger to GitHub Actions"
        )
        logger.info("Created IAM Role: %s", ROLE_NAME)
        role_arn = role["Role"]["Arn"]
        
        # Attach Basic Execution Policy
        iam.attach_role_policy(
            RoleName=ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        )
        time.sleep(10)  # Wait for IAM role propagation
    except iam.exceptions.EntityAlreadyExistsException:
        role = iam.get_role(RoleName=ROLE_NAME)
        role_arn = role["Role"]["Arn"]
        logger.info("Using existing IAM Role: %s", role_arn)

    # 2. Package Lambda code into zip
    import io, zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("lambda_function.py", LAMBDA_CODE)
    zip_bytes = zip_buffer.getvalue()

    # 3. Create or update Lambda function
    gh_token = "gho_dummy_token"  # GitHub token from env or secrets
    try:
        resp = lambda_cli.create_function(
            FunctionName=LAMBDA_NAME,
            Runtime="python3.12",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_bytes},
            Timeout=30,
            Environment={
                "Variables": {
                    "GITHUB_REPO": "priyanshu-projects/Movie_Recommender",
                    "GITHUB_TOKEN": gh_token
                }
            }
        )
        logger.info("✓ Lambda function created: %s", resp["FunctionArn"])
        fn_arn = resp["FunctionArn"]
    except lambda_cli.exceptions.ResourceConflictException:
        resp = lambda_cli.update_function_code(
            FunctionName=LAMBDA_NAME,
            ZipFile=zip_bytes
        )
        fn_arn = resp["FunctionArn"]
        logger.info("✓ Lambda function code updated.")

    logger.info("✓ AWS Lambda trigger deployment complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    deploy()
