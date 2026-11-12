"""Use the authors' current CSV converter, without its optional W&B upload."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.scripts.csv_to_npz import MotionLoader

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    out = ROOT / "data/lafan1/npz"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((ROOT / "data/lafan1/csv").glob("*.csv")):
        target = out / (path.stem + ".npz")
        motion = MotionLoader(str(path), input_fps=30.0, output_fps=50.0, device="cpu")
        arrays = motion.arrays()
        if any(
            not np.isfinite(v).all() for k, v in arrays.items() if k != "joint_names"
        ):
            raise ValueError(f"Nonfinite reference data: {path.name}")
        np.testing.assert_allclose(
            np.linalg.norm(arrays["root_quat_w"], axis=-1), 1, atol=1e-4
        )
        np.savez(target, **arrays)
        frames = len(arrays["joint_pos"])
        rows.append(
            {
                "name": path.stem,
                "frames": frames,
                "seconds": (frames - 1) / 50.0,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        print(f"Converted {path.name}: {frames} frames", flush=True)
    manifest = {
        "converter": "mjlab.scripts.csv_to_npz.MotionLoader.arrays",
        "input_fps": 30,
        "output_fps": 50,
        "count": len(rows),
        "frames": sum(r["frames"] for r in rows),
        "seconds": sum(r["seconds"] for r in rows),
        "clips": rows,
    }
    (ROOT / "data/lafan1/manifest_converted.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "clips"}), flush=True)


if __name__ == "__main__":
    main()
