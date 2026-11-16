"""Resume a verified BONES-SEED archive download using authenticated HF ranges.

Credentials and signed download URLs stay in memory. Completion requires the
publisher's LFS SHA-256. This is data preparation only.
"""

import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_url
from supermimic.utils.reproduction_paths import (
    data_root,
    portable_path,
)

ROOT = Path(__file__).resolve().parents[2] / "reproduction"
CHUNK = 4 * 1024 * 1024


def main():
    source = json.loads((ROOT / "setup/bones_access.json").read_text())
    assert source["access_verified"]
    out = data_root() / "boneseed/raw"
    out.mkdir(parents=True, exist_ok=True)
    final = out / "g1.tar.gz"
    partial = out / "g1.tar.gz.partial"
    progress = out / "download_progress.json"
    size = source["archive_bytes"]
    expected = source["lfs_sha256"]
    if (
        final.exists()
        and hashlib.file_digest(final.open("rb"), "sha256").hexdigest() == expected
    ):
        print("Archive already verified.", flush=True)
        return
    metadata_url = hf_hub_url(
        "bones-studio/seed",
        "g1.tar.gz",
        repo_type="dataset",
        revision=source["revision"],
    )
    location = get_hf_file_metadata(metadata_url, token=True).location
    done = set()
    if progress.exists() and partial.exists():
        state = json.loads(progress.read_text())
        if state.get("sha256") == expected and state.get("chunk_bytes") == CHUNK:
            done = set(state["completed_chunks"])
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    start_time = time.monotonic()
    previous = len(done)
    total = (size + CHUNK - 1) // CHUNK
    print(
        f"Archive {size / 1e9:.2f} GB, {total} chunks; resuming {len(done)} completed chunks",
        flush=True,
    )

    def save():
        temp = progress.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {
                    "sha256": expected,
                    "chunk_bytes": CHUNK,
                    "completed_chunks": sorted(done),
                    "total_chunks": total,
                }
            )
            + "\n"
        )
        temp.replace(progress)

    def fetch(index):
        nonlocal location
        start = index * CHUNK
        end = min(start + CHUNK, size) - 1
        for attempt in range(8):
            try:
                request = urllib.request.Request(
                    location, headers={"Range": f"bytes={start}-{end}"}
                )
                # HF metadata uses the configured proxy; its AWS CDN is much
                # faster over a direct connection on this server.
                opener = (
                    urllib.request.build_opener()
                    if os.environ.get("BONES_USE_PROXY") == "1"
                    else urllib.request.build_opener(urllib.request.ProxyHandler({}))
                )
                with opener.open(request, timeout=30) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{size}"
                    ):
                        raise ValueError(
                            "Server did not honor the requested byte range"
                        )
                    block = response.read()
                if len(block) != end - start + 1:
                    raise ValueError("Incomplete byte range")
                written = os.pwrite(fd, block, start)
                if written != len(block):
                    raise OSError("Short local write")
                with lock:
                    done.add(index)
                    if len(done) % 32 == 0:
                        save()
                        fetched = (len(done) - previous) * CHUNK
                        elapsed = time.monotonic() - start_time
                        print(
                            f"Verified ranges: {len(done)}/{total}, {(len(done) * CHUNK) / 1e9:.2f} GB, {fetched / max(elapsed, 1) / 1e6:.1f} MB/s",
                            flush=True,
                        )
                return
            except Exception as exc:
                if attempt == 7:
                    # Deliberately omit URL-bearing exception text.
                    raise RuntimeError(
                        f"Chunk {index} failed: {type(exc).__name__}"
                    ) from None
                if getattr(exc, "code", None) == 403:
                    with lock:
                        location = get_hf_file_metadata(
                            metadata_url, token=True
                        ).location
                time.sleep(min(1 + attempt, 5))

    try:
        pending = [i for i in range(total) if i not in done]
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            for _ in pool.map(fetch, pending):
                pass
    finally:
        with lock:
            save()
        os.close(fd)
    if len(done) != total:
        raise RuntimeError("Archive download is incomplete")
    actual = hashlib.file_digest(partial.open("rb"), "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Archive SHA-256 mismatch: {actual}")
    partial.replace(final)
    result = {
        "status": "verified",
        "path": portable_path(final),
        "bytes": size,
        "sha256": actual,
        "revision": source["revision"],
        "seconds_this_run": time.monotonic() - start_time,
    }
    (out / "verified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
