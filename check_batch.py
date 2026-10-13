"""
check_batch.py
--------------
Look up an existing OpenAI Batch job by ID (no need to resubmit anything),
print its status, and download whatever output/error files it has so far.

Usage:
    python check_batch.py batch_6a8ff73044708190be54f50b3cbd4965

    # optional: choose where to save the downloaded files
    python check_batch.py batch_6a8ff73044708190be54f50b3cbd4965 --output-dir results_openai/

Requires: pip install openai
Requires an API key: set OPENAI_API_KEY (or pass --api-key)
"""
import argparse
import sys
from pathlib import Path

from openai import OpenAI


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch_id", help="e.g. batch_6a8ff73044708190be54f50b3cbd4965")
    ap.add_argument("--output-dir", default=".", help="Where to save downloaded output/error files")
    ap.add_argument("--api-key", default=None, help="Defaults to OPENAI_API_KEY env var")
    args = ap.parse_args()

    client = OpenAI(api_key=args.api_key) if args.api_key else OpenAI()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = client.batches.retrieve(args.batch_id)

    print(f"Batch ID:        {batch.id}")
    print(f"Status:          {batch.status}")
    print(f"Endpoint:        {batch.endpoint}")
    print(f"Created at:      {batch.created_at}")
    print(f"Completed at:    {getattr(batch, 'completed_at', None)}")
    print(f"Failed at:       {getattr(batch, 'failed_at', None)}")
    print(f"Expired at:      {getattr(batch, 'expired_at', None)}")
    print(f"Request counts:  {batch.request_counts}")
    print(f"Metadata:        {batch.metadata}")
    print(f"Output file id:  {batch.output_file_id}")
    print(f"Error file id:   {batch.error_file_id}")
    if getattr(batch, "errors", None):
        print(f"Batch-level errors: {batch.errors}")

    if batch.status not in ("completed", "failed", "cancelled", "expired"):
        print(f"\nJob is still in progress (status: {batch.status}). Nothing to download yet.")
        return

    if batch.output_file_id:
        print(f"\nDownloading output file {batch.output_file_id} ...")
        content = client.files.content(batch.output_file_id)
        out_path = output_dir / "batch_results_raw.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content.text)
        print(f"Wrote: {out_path}")
        n_lines = content.text.count("\n") or (1 if content.text.strip() else 0)
        print(f"({n_lines} result line(s))")
    else:
        print("\nNo output file.")

    if batch.error_file_id:
        print(f"\nDownloading error file {batch.error_file_id} ...")
        error_content = client.files.content(batch.error_file_id)
        err_path = output_dir / "batch_errors_raw.jsonl"
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(error_content.text)
        print(f"Wrote: {err_path}")
        print("\n--- first errors ---")
        print(error_content.text[:3000])
    else:
        print("No error file.")


if __name__ == "__main__":
    main()