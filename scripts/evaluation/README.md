# 完整动作测评

本目录用于 SuperMimic 策略的完整动作测评，包括模型选优、单卡动作调度、逐动作计分、汇总和同步视频。下一阶段先筛选 BONES 并从头重训，再将新旧模型在同一筛选集上比较；已有 LAFAN1 模型只做推理核验。每个条件仅选一个最佳模型用于正式测评。实施顺序和预算见 [PLAN_12_16_0.md](../../PLAN_12_16_0.md)，当前尚未实现这些正式测评脚本。固定选优与推理工程参数见 [配置说明](../../docs/EXPERIMENT_CONFIG.md)。

已有 [eval_smoke.py](../validation/eval_smoke.py) 只检查有限步回放与模型加载，不能替代论文定义的动作评测。正式测评输出统一放在 `reproduction/evaluations/`，与训练和 smoke test 产物分开。
