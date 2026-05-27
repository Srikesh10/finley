import os
from dotenv import load_dotenv
from anthropic import AnthropicBedrock

load_dotenv()

MODEL          = "us.anthropic.claude-opus-4-6-v1"          # initial analysis (deep)
SONNET_MODEL   = "us.anthropic.claude-sonnet-4-6"            # chat advice / reasoning
HAIKU_MODEL    = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # chat data lookups
AWS_KEY        = os.getenv("AWS_KEY", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "")
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")


def make_client() -> AnthropicBedrock:
    return AnthropicBedrock(
        aws_access_key=AWS_KEY,
        aws_secret_key=AWS_SECRET_KEY,
        aws_region=AWS_REGION,
    )
