"""Create (but do not automatically start) a SageMaker Processing Job request.

Run this only after pushing the Docker image to ECR and uploading the JSON dataset
to S3. It requires AWS credentials with SageMaker, S3 and IAM pass-role access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import boto3


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit ClearCall offline evaluation to SageMaker Processing.")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--input-s3-uri", required=True, help="S3 URI of synthetic_conversations.json")
    parser.add_argument("--output-s3-uri", required=True, help="S3 prefix for evaluation artifacts")
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--llm-provider", default="azure_openai", choices=["ollama", "azure_openai"])
    parser.add_argument("--retrieval-provider", default="azure_ai_search", choices=["local", "azure_ai_search"])
    args = parser.parse_args()

    job_name = f"clearcall-evaluation-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    response = boto3.client("sagemaker", region_name=args.region).create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=args.role_arn,
        AppSpecification={
            "ImageUri": args.image_uri,
            "ContainerEntrypoint": ["python", "sagemaker/processing_entrypoint.py"],
        },
        ProcessingResources={"ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.large", "VolumeSizeInGB": 10}},
        ProcessingInputs=[{
            "InputName": "synthetic-conversations",
            "S3Input": {
                "S3Uri": args.input_s3_uri,
                "LocalPath": "/opt/ml/processing/input",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        }],
        ProcessingOutputConfig={"Outputs": [{
            "OutputName": "evaluation-artifacts",
            "S3Output": {"S3Uri": args.output_s3_uri, "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob"},
        }]},
        Environment={
            "MLFLOW_EXPERIMENT_NAME": "clearcall-compliance-evaluation",
            "CLEARCALL_LLM_PROVIDER": args.llm_provider,
            "CLEARCALL_RETRIEVAL_PROVIDER": args.retrieval_provider,
        },
    )
    print({"processing_job_name": job_name, "arn": response["ProcessingJobArn"]})


if __name__ == "__main__":
    main()
