# Porous Film Generator 2.0 外部交接包

本包用于代码审查、复现和二次开发。请先阅读 `docs/v2孔生成器说明.md`。

## 固定版本

- 正式名称：Porous Film Generator 2.0 — Complex Shapes
- Git 标签：`2.0`
- 维护分支：`release/2.0`
- 提交：`1c9a10793e96437202482ec44e4263b87ef64882`
- Git tree：`966a97eb087426885eb4eab825b021b31113b5c3`
- 源码内包版本：`0.4.0.dev1`
- 交接制作日期：2026-08-31（Asia/Shanghai）

## 建议阅读顺序

1. `docs/v2孔生成器说明.md`
2. `review/KNOWN-ISSUES.md`
3. `source/porous-film-generator-2.0/README.md`
4. `source/porous-film-generator-2.0/skills/porous-film-generator/`
5. `review/code-map.md`
6. `examples/configs/`
7. `examples/synthetic-validator-pass/`
8. `review/logs/`

## 快速安装

### 从源码

```powershell
cd .\source\porous-film-generator-2.0
uv sync --frozen --all-groups
uv run porous-film version
uv run ruff check .
uv run pytest -q
```

### 从 wheel

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install .\dist\porous_film_generator-0.4.0.dev1-py3-none-any.whl
.\.venv\Scripts\porous-film version
```

Linux 下把 `.venv\Scripts\...` 换成 `.venv/bin/...`。

## 重要说明

- `examples/visual-only-failed-audit/` 的三个 GLB 是形状观察样例，均未通过全部正式分布审核。
- `examples/synthetic-validator-pass/` 是测试构造的最小契约样例，不代表真实材料结构。
- 外部包不含服务器账号、密码、私钥、真实提交文件或固定远程入口。
- 标签 2.0 源码没有 LICENSE 文件。外部分发、再授权或商业使用前必须由项目所有者明确授权。
- 完整测试在本机得到 `441 passed, 6 failed, 5 skipped`；失败分类见主文档和 `review/KNOWN-ISSUES.md`。