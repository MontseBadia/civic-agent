"""Download a dated CIViC release snapshot and write manifest.json next to it.

Usage: python data/download_civic.py [--release 01-Nov-2026] [--out data/raw]
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://civicdb.org/downloads"
DEFAULT_RELEASE = "01-Nov-2026"

# Accepted (editor-reviewed) evidence and assertions only. No VCFs.
FILES = [
    "AcceptedClinicalEvidenceSummaries.tsv",
    "AcceptedAssertionSummaries.tsv",
    "MolecularProfileSummaries.tsv",
    "VariantSummaries.tsv",
    "FeatureSummaries.tsv",
    "VariantGroupSummaries.tsv",
]

CHUNK = 1 << 20  # 1 MB


def file_url(release: str, name: str) -> str:
    return f"{BASE_URL}/{release}/{release}-{name}"


def download(url: str, dest: Path) -> tuple[str, int]:
    """Stream url to dest, return (sha256 hex, size in bytes)."""
    req = urllib.request.Request(url, headers={"User-Agent": "civic-agent"})
    sha = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        while chunk := resp.read(CHUNK):
            sha.update(chunk)
            f.write(chunk)
            size += len(chunk)
    return sha.hexdigest(), size


def snapshot(release: str, out: Path) -> Path:
    final = out / release
    if final.exists():
        sys.exit(f"{final} already exists. Snapshots are not overwritten; delete it to re-download.")

    partial = out / f"{release}.partial"
    if partial.exists():
        shutil.rmtree(partial)  # leftover from a failed run
    partial.mkdir(parents=True)

    downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = []
    for name in FILES:
        url = file_url(release, name)
        filename = f"{release}-{name}"
        sha, size = download(url, partial / filename)
        files.append({"name": filename, "url": url, "bytes": size, "sha256": sha})
        print(f"{filename}  {size} bytes  {sha[:12]}")

    manifest = {
        "source": "CIViC",
        "release": release,
        "downloaded_at": downloaded_at,
        "license": "CC0 1.0",
        "files": files,
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    partial.rename(final)
    return final


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", default=DEFAULT_RELEASE)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    args = p.parse_args()
    print(f"saved to {snapshot(args.release, args.out)}")


if __name__ == "__main__":
    main()
