## CPU 运行

```bash
cd /home/tiger/project/porous-film-generator-2.0-gpu

POROUS_FILM_VOXEL_BACKEND=cpu \
uv run --project source/porous-film-generator-2.0 \
  porous-film generate-geometry \
  --config examples/configs/01-config.yaml \
  --result-root runs
```

## GPU 运行

```bash
cd /home/tiger/project/porous-film-generator-2.0-gpu

POROUS_FILM_VOXEL_BACKEND=cuda POROUS_FILM_CUDA_DEVICE=0 \
uv run --project source/porous-film-generator-2.0 --extra gpu \
  porous-film generate-geometry \
  --config examples/configs/01-config.yaml \
  --result-root runs
```

## 查看结果

生成结束后，将命令打印的结果目录保存为：

```bash
RUN=/打印出的绝对路径
```

- 浏览器：打开 `$RUN/outputs/visual-report/index.html` 查看交互式报告、候选搜索和性能。
- Blender：导入 `$RUN/outputs/semiconductor_solid_target.glb`，孔显示为空腔。
- MeshLab：打开 `$RUN/qa_export/final_surface.ply` 查看孔表面。
- 并行摘要：查看 `$RUN/reports/parallel-summary.md`。
- 并行原始数据：查看 `$RUN/analysis/parallel-summary.json`。
