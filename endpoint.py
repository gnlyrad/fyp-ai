import json

import boto3
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # must run BEFORE boto3.client()

client = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

response = client.invoke_model(
    modelId="apac.amazon.nova-lite-v1:0",
    body=json.dumps({
        "messages": [{
            "role": "user",
            "content": [{"text": "Tell me a short story about a robot."}]
        }],
        "inferenceConfig": {"maxTokens": 1024}
    })
)

result = json.loads(response["body"].read())
print(result["output"]["message"]["content"][0]["text"])
