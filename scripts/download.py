import argparse
import os
import time
from pathlib import Path

import requests
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_url

parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index", help="Hugging Face repository ID")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face token")

args = parser.parse_args()

def download_with_progress(repo_id: str, filename: str, save_path: str, repo_type: str = "dataset", token: str = None):
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)

    target_path = Path(save_path) / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    # If the final file already exists, skip it and clean stale partial files.
    if target_path.exists():
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"{filename}: already exists, skip")
        return

    base_headers = {}
    if token:
        base_headers["Authorization"] = f"Bearer {token}"

    max_retries = 20
    attempt = 0

    while True:
        resume_size = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = dict(base_headers)
        if resume_size > 0:
            headers["Range"] = f"bytes={resume_size}-"

        try:
            with requests.get(url, stream=True, headers=headers, timeout=(10, 600)) as response:
                # If server doesn't honor range request, restart this file from scratch.
                if response.status_code == 200 and resume_size > 0:
                    resume_size = 0

                response.raise_for_status()

                content_length = int(response.headers.get("Content-Length", 0))
                total_size = content_length + resume_size if content_length > 0 else None
                write_mode = "ab" if resume_size > 0 else "wb"

                with open(tmp_path, write_mode) as f, tqdm(
                    total=total_size,
                    initial=resume_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=filename,
                    mininterval=0.2,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))

            os.replace(tmp_path, target_path)
            return
        except requests.RequestException as e:
            attempt += 1
            if attempt >= max_retries:
                raise RuntimeError(f"Download failed for {filename} after {max_retries} retries") from e
            wait_seconds = min(30, attempt * 2)
            print(f"{filename}: network issue ({e}), retry {attempt}/{max_retries} in {wait_seconds}s")
            time.sleep(wait_seconds)


hf_token = args.token or os.getenv("HF_TOKEN")

repo_id = "PeterJinGo/wiki-18-e5-index"
for file in ["part_aa", "part_ab"]:
    download_with_progress(
        repo_id=repo_id,
        filename=file,
        save_path=args.save_path,
        repo_type="dataset",
        token=hf_token,
    )

repo_id = "PeterJinGo/wiki-18-corpus"
download_with_progress(
        repo_id=repo_id,
        filename="wiki-18.jsonl.gz",
        save_path=args.save_path,
        repo_type="dataset",
        token=hf_token,
)
