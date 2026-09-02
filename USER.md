# Porous Film Generator 2.0 使用说明

默认例子使用 CPU 运行，并生成自包含的静态 HTML 报告。

```bash
cd ~/project/porous-film-generator-2.0/source/porous-film-generator-2.0
uv sync --frozen --all-groups --python 3.12
./.venv/bin/porous-film generate-geometry \
  --config ../../examples/configs/quick-visual-demo.yaml \
  --result-root ../../runs \
  --no-parallel
```

可视化包含 Geometry、Validation、Optimization 和 Performance 四个页面。

```bash
cd ~/project/porous-film-generator-2.0/source/porous-film-generator-2.0
REPORT=$(find ../../runs -path '*/outputs/visual-report/index.html' -type f | sort | tail -1)
open "$REPORT"
```
