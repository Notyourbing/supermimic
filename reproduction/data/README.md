# 数据目录

Git 中保存下载来源、动作清单、转换摘要和 packed 数据校验；原始 CSV、NPZ、NPY 与归档由 `.gitignore` 排除。

| 子目录 | 训练数据 | 关键验证文件 |
|---|---|---|
| `lafan1/` | G1 重定向动作，40 条，50 Hz | `manifest_raw.json`、`manifest_converted.json` |
| `boneseed/` | BONES 原始公开集合，142,220 条，50 Hz | `conversion_summary.json`、`packed_unfiltered/verified.json` |

新服务器默认用 `bash reproduction/prepare_data.sh lafan` 或 `bones` 将完整数据准备到此目录。当前服务器的完整数据在 `/home/zhuobing/projects/supermimic/official_reproduction/data/`，训练配置中的路径保持不变。操作步骤见 [复现指南](../../docs/guides/reproduction.md)。
