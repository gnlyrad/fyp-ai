"""Fetch an AWS Transcribe output JSON from S3, for `ingest`.

Accepted URL shapes:
- `s3://bucket/key.json`
- virtual-hosted HTTPS: `https://bucket.s3.<region>.amazonaws.com/key.json`
  (or `bucket.s3.amazonaws.com`)
- path-style HTTPS: `https://s3.<region>.amazonaws.com/bucket/key.json` --
  the shape of Transcribe's own `TranscriptFileUri` for a customer bucket
- a presigned HTTPS URL (has an `X-Amz-Signature` query parameter): fetched
  with a plain GET, no AWS credentials needed, since the signature already
  carries the authorisation. This is also what Transcribe returns for a
  service-managed output bucket.

Every other shape is read with `boto3` `GetObject`, which uses the standard
AWS credential chain (env vars, `~/.aws`, an IAM role). A Bedrock bearer
token (`AWS_BEARER_TOKEN_BEDROCK`) does not authorise S3.

`boto3` is imported lazily so importing the package never needs it.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

DEFAULT_REGION = "ap-southeast-1"
# A Transcribe JSON is word-level, so long calls get big, but not this big.
MAX_BYTES = int(os.getenv("COMPLIANCE_S3_MAX_BYTES", str(50 * 1024 * 1024)))
HTTP_TIMEOUT_S = float(os.getenv("COMPLIANCE_S3_TIMEOUT_S", "30"))

# bucket.s3.amazonaws.com | bucket.s3.<region>.amazonaws.com | bucket.s3-<region>.amazonaws.com
_VIRTUAL_HOST_RE = re.compile(r"^(?P<bucket>.+)\.s3(?:[.-](?P<region>[a-z0-9-]+))?\.amazonaws\.com$")
# s3.amazonaws.com | s3.<region>.amazonaws.com | s3-<region>.amazonaws.com
_PATH_HOST_RE = re.compile(r"^s3(?:[.-](?P<region>[a-z0-9-]+))?\.amazonaws\.com$")


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str
    region: str | None = None


def is_presigned(url: str) -> bool:
    query = parse_qs(urlparse(url).query)
    return "X-Amz-Signature" in query or "Signature" in query


def parse_s3_url(url: str) -> S3Location:
    """Bucket, key and (when the host names one) region for an S3 URL.

    Raises `ValueError` for anything that isn't an S3 object URL.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme == "s3":
        bucket, key = parsed.netloc, unquote(parsed.path.lstrip("/"))
        region = None
    elif parsed.scheme in ("https", "http"):
        host = (parsed.hostname or "").lower()
        path = unquote(parsed.path.lstrip("/"))
        virtual = _VIRTUAL_HOST_RE.match(host)
        path_style = _PATH_HOST_RE.match(host)
        if path_style:  # checked first: "s3.<region>.amazonaws.com" also fits the virtual-host pattern
            bucket, _, key = path.partition("/")
            region = path_style.group("region")
        elif virtual:
            bucket, key = virtual.group("bucket"), path
            region = virtual.group("region")
        else:
            raise ValueError(f"not an S3 URL: {url!r}")
    else:
        raise ValueError(f"not an S3 URL: {url!r} (expected s3://bucket/key or an https S3 URL)")

    if not bucket or not key:
        raise ValueError(f"S3 URL must name a bucket and an object key: {url!r}")
    return S3Location(bucket=bucket, key=key, region=region)


def _get_via_boto3(location: S3Location) -> bytes:
    import boto3  # lazy

    region = location.region or os.getenv("AWS_REGION") or DEFAULT_REGION
    client = boto3.client("s3", region_name=region)
    response = client.get_object(Bucket=location.bucket, Key=location.key)
    if response.get("ContentLength", 0) > MAX_BYTES:
        raise ValueError(f"S3 object is larger than {MAX_BYTES} bytes")
    return response["Body"].read()


def _get_via_https(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as response:  # noqa: S310 - presigned S3 URL
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError(f"S3 object is larger than {MAX_BYTES} bytes")
    return body


def fetch_transcribe_json(url: str) -> dict:
    """Download and parse the Transcribe output at `url`.

    Returns the parsed JSON as-is: normally a full Transcribe job
    (`{"jobName": ..., "results": {...}}`), sometimes just its `results`
    object. Raises `ValueError` with a readable message on any failure
    (bad URL, missing object, access denied, no credentials, bad JSON).
    """
    try:
        if urlparse(url).scheme in ("https", "http") and is_presigned(url):
            body = _get_via_https(url)
        else:
            body = _get_via_boto3(parse_s3_url(url))
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - botocore/urllib errors -> one readable error
        raise ValueError(f"could not fetch transcript from {url!r}: {type(exc).__name__}: {exc}") from exc

    try:
        payload = json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"transcript at {url!r} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"transcript at {url!r} must be a JSON object")
    return payload
