"""Adapt the original BONES-SEED CSV schema to the official mjlab motion format.

Units / rotation convention follow NVIDIA's SEED G1 preparation documentation:
cm, degrees, extrinsic xyz Euler, 120 Hz. Resampling and velocities are computed
by the official mjlab MotionLoader. No motion quality/name filtering is implied.
The tar stream is processed without writing tens of GB of intermediate CSVs.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import multiprocessing
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from mjlab.scripts.csv_to_npz import JOINT_NAMES, MotionLoader
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2] / "reproduction"
VERSION = "bones-cm-deg-extrinsic-xyz-v1"


class BonesLoader(MotionLoader):
    def _load_motion(self):
        data = self._bones_data
        header = data.readline().decode().strip().split(",")
        expected = [
            "Frame",
            *(f"root_translate{s}" for s in "XYZ"),
            *(f"root_rotate{s}" for s in "XYZ"),
            *(name + "_dof" for name in JOINT_NAMES),
        ]
        if header != expected:
            raise ValueError("Unexpected BONES-SEED columns / joint ordering")
        values = np.loadtxt(data, delimiter=",", dtype=np.float64, ndmin=2)
        if (
            len(values) < 3
            or values.shape[1] != len(expected)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid or too-short BONES-SEED motion")
        if not np.allclose(np.diff(values[:, 0]), 1):
            raise ValueError("Nonconsecutive 120 Hz input frames")
        quat_xyzw = Rotation.from_euler("xyz", values[:, 4:7], degrees=True).as_quat()
        self.motion_base_poss_input = torch.from_numpy(
            (values[:, 1:4] * 0.01).astype(np.float32)
        )
        self.motion_base_rots_input = torch.from_numpy(
            quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
        )
        self.motion_dof_poss_input = torch.from_numpy(
            np.deg2rad(values[:, 7:]).astype(np.float32)
        )
        self.input_frames = len(values)
        self.duration = (self.input_frames - 1) * self.input_dt


def worker(task):
    name, data, destination = task
    torch.set_num_threads(1)
    source_sha = hashlib.sha256(data).hexdigest()
    target = Path(destination) / Path(name).relative_to("g1/csv").with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with np.load(target) as old:
            if (
                str(old.get("conversion_version", "")) == VERSION
                and str(old.get("source_sha256", "")) == source_sha
            ):
                return {
                    "source": name,
                    "output": str(target),
                    "frames": len(old["joint_pos"]),
                    "source_sha256": source_sha,
                    "cached": True,
                }
    loader = BonesLoader.__new__(BonesLoader)
    loader._bones_data = io.BytesIO(data)
    loader.__init__(name, input_fps=120.0, output_fps=50.0, device="cpu")
    arrays = loader.arrays()
    for key, value in arrays.items():
        if key != "joint_names" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite converted channel {key}: {name}")
    np.testing.assert_allclose(
        np.linalg.norm(arrays["root_quat_w"], axis=-1), 1, atol=1e-4
    )
    temp = target.with_suffix(".part")
    with temp.open("wb") as fh:
        np.savez(fh, **arrays, source_sha256=source_sha, conversion_version=VERSION)
    temp.replace(target)
    return {
        "source": name,
        "output": str(target),
        "frames": len(arrays["joint_pos"]),
        "source_sha256": source_sha,
        "cached": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=ROOT / "data/boneseed/raw/g1.tar.gz")
    p.add_argument("--output", type=Path, default=ROOT / "data/boneseed/npz")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_name = "conversion_smoke" if args.limit else "conversion_full_unfiltered"
    manifest_path = args.output.parent / (result_name + ".json")
    rows = []
    pending = set()
    errors = []
    started = time.monotonic()

    def collect(future):
        try:
            rows.append(future.result())
        except Exception as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc)})
        if (len(rows) + len(errors)) % 1000 == 0:
            print(
                f"Converted {len(rows)}, errors {len(errors)}, {time.monotonic() - started:.1f}s",
                flush=True,
            )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        with tarfile.open(args.archive, mode="r|gz") as archive:
            submitted = 0
            for member in archive:
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                if (
                    not member.name.startswith("g1/csv/")
                    or ".." in Path(member.name).parts
                ):
                    raise ValueError("Unexpected archive member path")
                data = archive.extractfile(member).read()
                pending.add(pool.submit(worker, (member.name, data, str(args.output))))
                submitted += 1
                if len(pending) >= args.workers * 3:
                    done, pending = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        collect(future)
                if args.limit and submitted >= args.limit:
                    break
            for future in concurrent.futures.as_completed(pending):
                collect(future)
    frames = sum(row["frames"] for row in rows)
    result = {
        "status": "passed" if not errors else "failed",
        "filtered": False,
        "conversion_version": VERSION,
        "archive": str(args.archive),
        "count": len(rows),
        "frames": frames,
        "seconds": (frames - len(rows)) / 50.0,
        "elapsed_seconds": time.monotonic() - started,
        "errors": errors,
        "source_convention": "https://github.com/NVlabs/ProtoMotions/blob/main/docs/source/getting_started/seed_g1_csv_preparation.rst",
        "resampling_and_velocities": "official mjlab.scripts.csv_to_npz.MotionLoader",
        "clips": sorted(rows, key=lambda row: row["source"]),
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "clips"}, indent=2
        ),
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
