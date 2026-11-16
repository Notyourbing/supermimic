# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## 本 fork 的复现入口

本项目复现 SuperMimic 的 G1 动作模仿方法。下一阶段先筛选 BONES 并在单卡上从头重训，再将新旧模型放到同一筛选集上比较；已有 LAFAN1 模型用于补齐论文指标。任务按优先级执行，详见下方计划。

| 要做的事 | 入口 |
|---|---|
| 阅读训练结果和图表 | [RESULT.md](RESULT.md) |
| 查看固定超参数与实现设置 | [配置说明](docs/EXPERIMENT_CONFIG.md) |
| 查找当前模型与训练统计 | [实验入口](reproduction/experiments/README.md) |
| 查看筛选、单卡重训与测评计划 | [PLAN_12_16_0.md](PLAN_12_16_0.md) |
| 安装环境、准备数据或迁移服务器 | [复现指南](docs/guides/reproduction.md) |
| 了解目录与产物位置 | [项目结构](docs/PROJECT_STRUCTURE.md) |
| 查找可执行脚本 | [脚本索引](scripts/README.md) |
| 查阅历史训练批次 | [训练批次](reproduction/campaigns/20261114_8xa100/README.md) |

`src/` 是上游算法实现，`scripts/` 是按用途分类的复现工具，`docs/` 保存文档与图表，`reproduction/` 保存环境入口、数据清单和实验产物。本 fork 的环境请使用复现指南中的固定依赖安装流程；下方保留上游的使用说明。

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
```

## Usage

The task ID and fixed implementation settings used in this reproduction are documented in [EXPERIMENT_CONFIG.md](docs/EXPERIMENT_CONFIG.md).

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
