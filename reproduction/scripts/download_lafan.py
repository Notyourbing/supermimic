"""Fetch the public G1 LAFAN1 retargets and verify the upstream Git blob IDs."""

import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"


def download(entry):
    path = ROOT / "data/lafan1/csv" / Path(entry["path"]).name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{entry['path']}"
    for attempt in range(4):
        try:
            data = (
                path.read_bytes()
                if path.exists()
                else urllib.request.urlopen(url, timeout=45).read()
            )
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if len(data) != entry["size"] or blob != entry["oid"]:
                raise ValueError(f"Upstream blob mismatch: {path.name}")
            temp = path.with_suffix(".part")
            temp.write_bytes(data)
            temp.replace(path)
            result = dict(
                entry, local=str(path), sha256=hashlib.sha256(data).hexdigest(), url=url
            )
            print(f"verified {path.name}: {len(data)} bytes", flush=True)
            return result
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def main():
    entries = json.loads((ROOT / "setup/lafan_tree_proxy.json").read_text())
    entries = [e for e in entries if e["type"] == "file" and e["path"].endswith(".csv")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(download, entries))
    manifest = {
        "repository": REPO,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(files),
        "files": files,
    }
    (ROOT / "data/lafan1/manifest_raw.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"All {len(files)} clips verified.", flush=True)


if __name__ == "__main__":
    main()
