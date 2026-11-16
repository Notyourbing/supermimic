# 完整动作测评

本目录用于下一阶段的动作清单生成、逐动作完整测评、分片调度、汇总和同步视频工具，实施计划见 [PLAN_12_16_0.md](../../PLAN_12_16_0.md)。当前尚未实现这些正式测评脚本。

已有 [eval_smoke.py](../validation/eval_smoke.py) 只检查有限步回放与模型加载，不能替代论文定义的动作评测。正式测评输出统一放在 `reproduction/evaluations/`，与训练和 smoke test 产物分开。
