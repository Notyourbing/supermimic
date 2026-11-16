"""A tiny eight-device communication check, with no model training."""

import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

root = Path(__file__).resolve().parents[2] / "reproduction"
rank = int(os.environ["LOCAL_RANK"])
torch.set_num_threads(1)
torch.cuda.set_device(rank)
dist.init_process_group(
    "nccl",
    timeout=datetime.timedelta(seconds=60),
    device_id=torch.device(f"cuda:{rank}"),
)
x = torch.full((1024,), rank + 1.0, device=f"cuda:{rank}")
dist.all_reduce(x)
torch.cuda.synchronize()
expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
assert torch.all(x == expected), (rank, x)
a = torch.randn(256, 256, device=f"cuda:{rank}", dtype=torch.bfloat16)
b = a @ a.T
assert torch.isfinite(b).all()
info = {
    "rank": rank,
    "name": torch.cuda.get_device_name(rank),
    "compute_capability": torch.cuda.get_device_capability(rank),
    "allreduce_sum": expected,
    "bf16_matmul_finite": True,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "nccl": torch.cuda.nccl.version(),
    "free_bytes": torch.cuda.mem_get_info(rank)[0],
    "status": "passed",
}
out = root / "smoke_tests/hardware"
out.mkdir(parents=True, exist_ok=True)
(out / f"gpu{rank}.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info), flush=True)
dist.barrier()
dist.destroy_process_group()
