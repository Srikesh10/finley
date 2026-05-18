import os
from dotenv import load_dotenv
from anthropic import AnthropicBedrock

load_dotenv()

MODEL          = "us.anthropic.claude-opus-4-6-v1"
AWS_KEY        = os.getenv("AWS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")


def make_client() -> AnthropicBedrock:
    return AnthropicBedrock(
        aws_access_key=AWS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        aws_region=AWS_REGION,
    )
