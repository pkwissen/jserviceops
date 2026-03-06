"""
Scan OpenSearch index and update documents' `document_title` by extracting KB info from text.
Run with Python in the project's virtualenv.

Example:
    python search/reindex_document_titles.py --dry-run
"""
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

from ..ingestion.kb_extractor import extract_kb_info
from ..config.settings import OPENSEARCH_ENDPOINT, AWS_REGION, INDEX_NAME

load_dotenv()


def create_client():
    if not OPENSEARCH_ENDPOINT:
        raise ValueError("OPENSEARCH_ENDPOINT not set in config/settings or env")
    host = OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", "")

    # Use AWS auth if credentials present in env or settings
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = AWS_REGION or os.getenv("AWS_REGION", "us-east-1")

    if aws_access_key and aws_secret:
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret,
            region_name=region,
        )
        credentials = session.get_credentials()
        auth = AWSV4SignerAuth(credentials, region, 'es')
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60,
        )
    else:
        # Try basic auth from env
        user = os.getenv("OPENSEARCH_USERNAME")
        pwd = os.getenv("OPENSEARCH_PASSWORD")
        if not user or not pwd:
            raise ValueError("No AWS creds or basic auth found for OpenSearch")
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=(user, pwd),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60,
        )
    return client


def reindex_titles(dry_run=True, batch_size=500):
    client = create_client()
    if not client.indices.exists(index=INDEX_NAME):
        print(f"Index {INDEX_NAME} does not exist")
        return

    total_updated = 0
    total_checked = 0

    # Initial search with scroll
    resp = client.search(index=INDEX_NAME, body={"query": {"match_all": {}}}, size=batch_size, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    hits = resp.get('hits', {}).get('hits', [])

    while hits:
        for hit in hits:
            total_checked += 1
            _id = hit.get('_id')
            src = hit.get('_source', {})
            current_title = src.get('document_title') or ''
            if current_title and 'unknown' not in current_title.lower() and 'kb_number not found' not in current_title.lower():
                # already has a reasonable title
                continue

            text = src.get('text', '')
            if not text:
                continue

            kb_info = extract_kb_info(text, src.get('document_key', ''))
            kb_title = kb_info.get('kb_title')
            kb_number = kb_info.get('kb_number')

            if kb_title and kb_title.lower() != 'unknown' and kb_number and kb_number.lower() != 'unknown':
                print(f"[{_id}] -> set document_title = {kb_title}")
                if not dry_run:
                    try:
                        client.update(index=INDEX_NAME, id=_id, body={"doc": {"document_title": kb_title}})
                        total_updated += 1
                    except Exception as e:
                        print(f"Failed to update {_id}: {e}")
        # fetch next batch
        if not scroll_id:
            break
        resp = client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = resp.get('_scroll_id')
        hits = resp.get('hits', {}).get('hits', [])

    print(f"Checked {total_checked} docs; updated {total_updated} docs (dry_run={dry_run})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Show changes without updating')
    parser.add_argument('--batch-size', type=int, default=500)
    args = parser.parse_args()

    reindex_titles(dry_run=args.dry_run, batch_size=args.batch_size)
