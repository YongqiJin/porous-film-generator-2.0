# 复杂孔形状生成 v1 实施计划

**设计规格：** `docs/superpowers/specs/2026-08-20-complex-shape-generation-v1-design.md`

1. 从 `original-v0.2.0` 建立 `experiment/complex-shape-generation-v1` 独立 worktree，验证基线测试。
2. 以 TDD 新建 `geometry/complex_shapes.py`，实现多叶 profile、变截面通道 profile、固定 Sobol 体积估计和自交诊断。
3. 扩展 `CompactUnit`、`ChannelUnit`、`build_units()` 和缩放函数；手工简单构造器保持 v1。
4. 为通道实现 profile 结点插入、段半径/AABB 缓存、暴力 SDF 参考和误差受控裁剪。
5. 将 `unit_geometry.jsonl`、`channel_curves.h5` 和 `pore_geometry.h5` 升为几何 schema v2。
6. 扩展独立 validator，同时保留 v1，独立重算 v2 体积、`eta/tau` 与形状合法性。
7. 扩展 `main_unit_metrics.csv` 和 optimizer realized JSON 的 `shape_complexity_summary`。
8. 将实验包版本改为 `0.3.0.dev1`，更新仓库 Skill 和输出文档，不改全局 Skill/稳定部署。
9. 运行 Ruff、完整 pytest、构建 wheel/sdist、SDF 等价测试和基线性能对照。
10. 在 `C:\Calculation_results` 生成 compact/channel/mixed 对照，运行独立 validator 和独立 subagent 审核。
11. 审查并提交实验分支；等待用户确认形状和统计后再讨论部署或合并。
