import os
from dotenv import load_dotenv

load_dotenv()

_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "")
_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_HOST   = os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))

enabled = False
_client = None

if _SECRET and _PUBLIC:
    try:
        from langfuse import Langfuse
        _client = Langfuse(secret_key=_SECRET, public_key=_PUBLIC, host=_HOST)
        enabled = _client.auth_check()
    except Exception:
        enabled = False


def log(name: str, model: str, input_msgs: list, output: str,
        input_tokens: int = 0, output_tokens: int = 0,
        metadata: dict = None):
    if not enabled:
        return
    try:
        with _client.start_as_current_observation(
            name=name, as_type="generation",
            input=input_msgs, output=output, model=model,
            usage_details={"input": input_tokens, "output": output_tokens},
            metadata=metadata or {},
        ):
            pass
    except Exception:
        pass


def score(name: str, value: float):
    if not enabled:
        return
    try:
        _client.score_current_trace(name=name, value=value)
    except Exception:
        pass


def flush():
    if enabled:
        try:
            _client.flush()
        except Exception:
            pass
