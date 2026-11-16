# 环境与安装证据

- `resolved_constraints.txt`：环境安装使用的精确依赖约束。
- `source_manifest.json`、`versions.json`：配套源码和版本记录。
- `pip-freeze.txt`、`conda-explicit.txt`：版本审计快照（本地安装引用已改为仓库相对路径）；正式安装仍使用固定约束与 bootstrap 入口。
- `bones_access.json`、`lafan_tree_proxy.json`：历史数据访问及下载来源记录；不替代新服务器的权限检查。
- `gpu_baseline_after_tests.json`、`migration_validation.json`：原服务器验证证据。
- `path_portability.json`：历史配置与清单中路径规范化的范围、原文件哈希和字段数量；未改写二进制模型。
- `wheel_manifest*.json`：安装传输审计，wheel 本体未纳入 Git。

安装入口为仓库根目录下的 `bash reproduction/bootstrap.sh`。已有 wheel、本地 Conda 缓存和配套依赖源码时，可使用 `bash reproduction/setup/restore_environment.sh` 离线恢复。完整说明见 [复现指南](../../docs/guides/reproduction.md)。
