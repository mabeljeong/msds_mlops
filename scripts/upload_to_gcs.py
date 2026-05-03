"""
Upload files from data/ to a Google Cloud Storage bucket.

Usage:
    python scripts/upload_to_gcs.py --bucket my-bucket --prefix rentiq/data/
"""

import argparse
from pathlib import Path

from google.cloud import storage

DATA_DIR = Path(__file__).parents[1] / "data"


def upload_file(bucket: storage.Bucket, local_path: Path, gcs_prefix: str) -> None:
    blob_name = gcs_prefix + local_path.name
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded {local_path} → gs://{bucket.name}/{blob_name}")


def upload_directory(bucket_name: str, local_dir: Path, prefix: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    files = list(local_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    for f in files:
        rel = f.relative_to(local_dir.parent)
        blob_name = prefix + str(rel)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(f))
        print(f"Uploaded {f} → gs://{bucket_name}/{blob_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload data/ to GCS")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--prefix", default="", help="GCS key prefix")
    parser.add_argument(
        "--subdir",
        default="raw",
        choices=["raw", "processed", "features", "scraped"],
        help="Subdirectory of data/ to upload",
    )
    args = parser.parse_args()

    local_dir = DATA_DIR / args.subdir
    upload_directory(args.bucket, local_dir, args.prefix)
