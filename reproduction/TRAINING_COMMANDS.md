# 正式训练命令（尚未执行）

先阅读 [交接说明](README.md) 和 [下一步](NEXT_STEPS.md)，完成新服务器的验证与资源协调。下面使用当前官方代码更新逻辑、固定种子 42、传感器观测/历史 10、默认 batch，显式限定 32 万收集循环。

在仓库根目录执行：

```bash
export REPO="$PWD"
CUDA_VISIBLE_DEVICES=0 bash reproduction/run_env.sh \
  python src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --motion-file "$REPO/reproduction/data/lafan1/packed" \
  --num-envs 4096 --env.seed 42 \
  --runner.actor-obs proprio --runner.history-length 10 \
  --runner.wm.batch-size 2048 --runner.policy.batch-size 4096 \
  --runner.max-iterations 320000 --runner.run-name official_gencoord_lafan42
```

MaxCoord 把任务 ID 换为 `SuperMimic-MaxCoord-G1`，run name 换为 `official_maxcoord_lafan42`，并选择可用显卡即可。两个任务 ID 的同样参数均已用 `--help` 验证解析，正式训练没有执行。

全量原始 BONES 路径为 `$REPO/reproduction/data/boneseed/packed_unfiltered`。它已经过原服务器全库格式校验与短时更新，但不是论文过滤集；严格论文指标对照前先补齐筛选协议。

候选 `--runner.wm.batch-size 4096 --runner.policy.batch-size 8192` 通过短测，具有不同的优化器更新频率，不替代默认效果基线。
