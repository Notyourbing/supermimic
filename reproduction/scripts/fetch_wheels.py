"""Download the resolved official PyPI wheels with verified HTTP ranges.

The server's proxy stalls long streamed wheel responses. Bounded ranges avoid
that transport issue; final bytes are checked against PyPI's SHA-256 digest.
"""

import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "setup/wheels"
CHUNK = 1024 * 1024


def request(url, headers=None):
    for attempt in range(4):
        try:
            opener = (
                urllib.request.build_opener(urllib.request.ProxyHandler({}))
                if "files.pythonhosted.org" in url
                else urllib.request.build_opener()
            )
            with opener.open(
                urllib.request.Request(url, headers=headers or {}), timeout=20
            ) as response:
                return response.read(), dict(response.headers)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 + attempt)


def metadata(url):
    name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    package, version = name.split("-")[:2]
    data, _ = request(f"https://pypi.org/pypi/{package}/{version}/json")
    info = json.loads(data)
    match = next(row for row in info["urls"] if row["filename"] == name)
    return {
        "filename": name,
        "url": match["url"],
        "size": match["size"],
        "sha256": match["digests"]["sha256"],
    }


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    log = (ROOT / "logs/install.log").read_text()
    urls = sorted(
        set(
            re.findall(
                r"Sending fresh GET request for: (https://\S+?\.whl)(?:\.metadata)?\s",
                log,
            )
        )
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        manifest = list(pool.map(metadata, urls))
    (ROOT / "setup/wheel_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    files = []
    for row in manifest:
        path = OUT / row["filename"]
        if (
            path.exists()
            and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        ):
            continue
        temp = path.with_suffix(".whl.part")
        fd = os.open(temp, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, row["size"])
        files.append((row, path, temp, fd))
    print(
        f"Fetching {len(files)} wheels, {sum(r['size'] for r, *_ in files) / 1e9:.2f} GB",
        flush=True,
    )
    done_bytes = 0
    lock = threading.Lock()

    def chunk(item):
        nonlocal done_bytes
        row, path, temp, fd, start = item
        end = min(start + CHUNK, row["size"]) - 1
        data, headers = request(row["url"], {"Range": f"bytes={start}-{end}"})
        if len(data) != end - start + 1:
            raise ValueError(f"Bad range size {path.name} {start}: {len(data)}")
        content_range = next(
            (v for k, v in headers.items() if k.lower() == "content-range"), ""
        )
        if row["size"] > CHUNK and not content_range.startswith(
            f"bytes {start}-{end}/"
        ):
            raise ValueError(f"Bad Content-Range: {content_range}")
        os.pwrite(fd, data, start)
        with lock:
            done_bytes += len(data)
            if done_bytes // (128 * 1024 * 1024) != (done_bytes - len(data)) // (
                128 * 1024 * 1024
            ):
                print(f"Downloaded {done_bytes / 1e9:.2f} GB", flush=True)

    tasks = []
    for row, path, temp, fd in sorted(files, key=lambda item: item[0]["size"]):
        for start in range(0, row["size"], CHUNK):
            length = min(CHUNK, row["size"] - start)
            previous = os.pread(fd, length, start)
            # Only complete, validated responses are written to the sparse file.
            # Final full-file SHA-256 remains authoritative on every resumed run.
            if previous.count(0) == length:
                tasks.append((row, path, temp, fd, start))
    print(f"Pending 1 MiB ranges: {len(tasks)}", flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            for _ in pool.map(chunk, tasks):
                pass
    finally:
        for _, _, _, fd in files:
            os.close(fd)
    for row, path, temp, _ in files:
        h = hashlib.file_digest(temp.open("rb"), "sha256").hexdigest()
        if h != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch {path.name}: {h}")
        temp.replace(path)
        print(f"Verified {path.name}", flush=True)
    print("All wheel hashes verified.", flush=True)


if __name__ == "__main__":
    main()
