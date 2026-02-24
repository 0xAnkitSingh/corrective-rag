"""Provision an Amazon Bedrock Knowledge Base with S3 data source.

Downloads the AWS-provided KB helper utility, creates an S3 bucket,
uploads documents, creates the knowledge base, runs ingestion, and
writes the resulting KB ID to the .env file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import boto3
import requests

logger = logging.getLogger(__name__)

_KB_HELPER_URL = (
    "https://raw.githubusercontent.com/aws-samples/amazon-bedrock-samples/"
    "main/rag/knowledge-bases/features-examples/utils/knowledge_base.py"
)


def download_kb_helper(target_path: str = "utils/knowledge_base.py") -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    resp = requests.get(_KB_HELPER_URL, timeout=60)
    resp.raise_for_status()
    with open(target_path, "w", encoding="utf-8") as fh:
        fh.write(resp.text)
    logger.info("Downloaded KB helper -> %s", target_path)


def create_s3_bucket(bucket_name: str, region: str) -> None:
    s3 = boto3.client("s3", region_name=region)
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region},
        )
    logger.info("Created S3 bucket: %s", bucket_name)


def upload_directory(local_dir: str, bucket_name: str, region: str) -> int:
    s3 = boto3.client("s3", region_name=region)
    count = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            filepath = os.path.join(root, fname)
            key = os.path.relpath(filepath, local_dir).replace("\\", "/")
            s3.upload_file(filepath, bucket_name, key)
            logger.info("Uploaded %s -> s3://%s/%s", filepath, bucket_name, key)
            count += 1
    return count


def upsert_env_file(values: dict[str, str], env_path: str = ".env") -> None:
    """Merge *values* into an existing .env file (or create one)."""
    lines: dict[str, str] = {}
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if "=" in stripped and not stripped.startswith("#"):
                    key, _, val = stripped.partition("=")
                    lines[key.strip()] = val.strip()
    lines.update(values)
    with open(env_path, "w", encoding="utf-8") as fh:
        for key, val in lines.items():
            fh.write(f"{key}={val}\n")
    logger.info("Updated %s", env_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Provision Amazon Bedrock Knowledge Base")
    p.add_argument(
        "--documents-dir",
        default="./onboarding_files",
        help="Local directory containing documents to ingest (default: ./onboarding_files)",
    )
    p.add_argument("--bucket-prefix", default="bedrock-hr-agent")
    p.add_argument("--kb-name-prefix", default="hr-agent-knowledge-base")
    p.add_argument("--region", default=None, help="AWS region (default: from boto3 session)")
    p.add_argument("--env-file", default=".env", help="Path to .env file to update")
    p.add_argument(
        "--test-query",
        default="Who is the medical insurance provider name?",
        help="Query to test the KB after creation",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )

    session = boto3.session.Session()
    region = args.region or session.region_name or "us-east-1"
    suffix = time.strftime("%Y%m%d%H%M%S")[-7:]

    bucket_name = f"{args.bucket_prefix}-{suffix}-bucket"
    kb_name = f"{args.kb_name_prefix}-{suffix}"

    # --- 1. Download KB helper utility ---
    download_kb_helper()

    sys.path.insert(0, ".")
    from utils.knowledge_base import BedrockKnowledgeBase  # type: ignore[import-untyped]

    # --- 2. S3 bucket + documents ---
    create_s3_bucket(bucket_name, region)
    n = upload_directory(args.documents_dir, bucket_name, region)
    logger.info("Uploaded %d files to %s", n, bucket_name)

    # --- 3. Knowledge Base ---
    logger.info("Creating Knowledge Base: %s", kb_name)
    kb = BedrockKnowledgeBase(
        kb_name=kb_name,
        kb_description="Knowledge Base containing onboarding and benefits documentation.",
        data_sources=[{"type": "S3", "bucket_name": bucket_name}],
        chunking_strategy="FIXED_SIZE",
        suffix=f"{suffix}-f",
    )

    logger.info("Waiting 30 s for KB to become available...")
    time.sleep(30)
    kb.start_ingestion_job()
    kb_id = kb.get_knowledge_base_id()
    logger.info("Knowledge Base ready — ID: %s", kb_id)

    # --- 4. Persist to .env ---
    upsert_env_file({"KNOWLEDGE_BASE_ID": kb_id, "AWS_REGION": region}, args.env_file)

    # --- 5. Smoke test ---
    client = boto3.client("bedrock-agent-runtime", region_name=region)
    response = client.retrieve_and_generate(
        input={"text": args.test_query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": (
                    f"arn:aws:bedrock:{region}::foundation-model/amazon.nova-micro-v1:0"
                ),
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": 5}
                },
            },
        },
    )
    logger.info("Test response: %s", response["output"]["text"])

    print(f"\nKnowledge Base ID : {kb_id}")
    print(f"Region            : {region}")
    print(f"Saved to          : {args.env_file}")


if __name__ == "__main__":
    main()
