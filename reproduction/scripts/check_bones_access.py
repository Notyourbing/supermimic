"""Verify THIS server's access without logging a token or signed URL."""

import json
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

root = Path(__file__).resolve().parents[1]
path = root / "setup/bones_access.json"
expected = json.loads(path.read_text())
revision = expected["revision"]
info = HfApi().repo_info(
    "bones-studio/seed", repo_type="dataset", revision=revision, files_metadata=True
)
item = next(item for item in info.siblings if item.rfilename == "g1.tar.gz")
metadata = get_hf_file_metadata(
    hf_hub_url(
        "bones-studio/seed", "g1.tar.gz", repo_type="dataset", revision=revision
    ),
    token=True,
)
assert metadata.size == expected["archive_bytes"]
assert item.lfs.sha256 == expected["lfs_sha256"]
print("Authenticated access verified for the pinned BONES-SEED G1 archive.")
