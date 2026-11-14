# 本次 smoke test 实测

独占结果已确认：False。全八组通过：False。

如果独占未确认，下表仅用于集成验证及共享负载观察，不能据此声称本项目达到 80% GPU 使用率或据此作准确独占工时预测。

| GPU | Run | 秒/循环 | transitions/s | Torch reserved GiB | 预算估时 GPU-h |
|---|---|---:|---:|---:|---:|
| 0 | exclusive_20261114_lafan1_gencoord_x1 | 0.225 | 18239 | 18.60 | 20.0 |
| 1 | exclusive_20261114_lafan1_maxcoord_x1 | 0.283 | 14449 | 21.07 | 25.2 |
| 2 | exclusive_20261114_boneseed_gencoord_x1 | 0.219 | 18722 | 32.42 | 19.4 |
| 3 | exclusive_20261114_boneseed_maxcoord_x1 | 0.506 | 8101 | 34.62 | 44.9 |
| 4 | exclusive_20261114_lafan1_gencoord_x2 | 0.359 | 22809 | 36.54 | 16.0 |
| 5 | exclusive_20261114_lafan1_maxcoord_x2 | failed/cancelled | | | |
| 6 | exclusive_20261114_boneseed_gencoord_x2 | failed/cancelled | | | |
| 7 | exclusive_20261114_boneseed_maxcoord_x2 | failed/cancelled | | | |

编译预热和末尾保存已剔除。预计值还需增加约 20% 运行余量；未收敛阶段的吞吐可能随训练状态变化。BONES 为未筛选数据，不等同于论文集合。
