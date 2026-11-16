"""Finish and verify data packaging after conversion; never starts training."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from supermimic.utils.reproduction_paths import (
    data_root,
)

ROOT = Path(__file__).resolve().parents[2] / "reproduction"
source_manifest = data_root() / "boneseed/conversion_full_unfiltered.json"
deadline = time.monotonic() + 3600
while not source_manifest.exists():
    if time.monotonic() > deadline:
        raise TimeoutError("Full data conversion did not finish within an hour")
    time.sleep(5)
source = json.loads(source_manifest.read_text())
if source["status"] != "passed" or source["errors"]:
    raise ValueError("Conversion has errors; refusing to package incomplete data")
output = data_root() / "boneseed/packed_unfiltered"
command = [
    "bash",
    str(ROOT / "run_env.sh"),
    "python",
    "-m",
    "mjlab.scripts.pack_motions",
    "--input",
    str(data_root() / "boneseed/npz/*/*.npz"),
    "--output",
    str(output),
    "--workers",
    "32",
]
started = time.monotonic()
subprocess.run(command, check=True)
ends = np.load(output / "motion_ends.npy")
if (
    len(ends) != source["count"]
    or int(ends[-1]) != source["frames"]
    or not (np.diff(ends) > 0).all()
):
    raise AssertionError("Packed clip boundaries do not match conversion manifest")
fields = {
    "joint_pos": 29,
    "joint_vel": 29,
    "root_pos_w": 3,
    "root_quat_w": 4,
    "root_lin_vel_w": 3,
    "root_ang_vel_w": 3,
}
files = {}
for name, width in fields.items():
    path = output / f"{name}.npy"
    arr = np.load(path, mmap_mode="r")
    if arr.shape != (source["frames"], width) or arr.dtype != np.float32:
        raise AssertionError(f"Unexpected packed array: {name} {arr.shape} {arr.dtype}")
    for offset in range(0, len(arr), 1_000_000):
        block = arr[offset : offset + 1_000_000]
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite packed array {name}")
        if name == "root_quat_w":
            np.testing.assert_allclose(np.linalg.norm(block, axis=-1), 1, atol=1e-4)
    files[path.name] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "bytes": path.stat().st_size,
        "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    }
result = {
    "status": "verified",
    "filtered": False,
    "count": source["count"],
    "frames": source["frames"],
    "seconds": source["seconds"],
    "hours": source["seconds"] / 3600,
    "resident_float32_bytes": source["frames"] * 71 * 4,
    "pack_and_verify_seconds": time.monotonic() - started,
    "files": files,
    "note": "Full raw release converted with the official resampler. This is not the paper's 129514-clip filtered benchmark.",
}
(output / "verified.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
