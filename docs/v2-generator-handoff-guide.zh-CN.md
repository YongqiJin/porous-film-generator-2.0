# v2孔生成器说明

> **用途：**本文件是 Porous Film Generator 2.0 的外部代码审查与二次开发主文档。它同时说明研究参数、实际输入、生成算法、最终结构审核、安装运行、输出文件、已知缺口和建议改造顺序。 **适用版本：**官方版本 2.0；Git 标签 2.0；维护分支 release/2.0；提交 1c9a10793e96437202482ec44e4263b87ef64882；源码内历史包版本 0.4.0.dev1。 **核对日期：**2026年8月31日（Asia/Shanghai）。

## 交接范围与一句话结论

Porous Film Generator 2.0 是一个基于“离散孔生成单元 + 有符号距离场 + 体素化”的三维薄膜孔结构生成器。它在原始版本的基础上增加了多叶紧凑孔、三维多弯变截面通道，并把主要审核改成从最终二值孔相重新测量，而不是直接相信生成时抽到的参数。

- 它是什么：可复现的 CPU 科学几何生成与审核程序，支持多候选、多随机种子、GLB/HDF5/CSV/JSON 输出和独立验证器。
- 它不是什么：不是 v3 的原生连续相场孔网络生成器；不是贝叶斯优化器；不是 GROMACS、Packmol 或输运计算程序；当前代码也不使用 GPU 加速。
- 正式推荐工作流：只给真实 target_box_A，生成孔相后从 final_phase.h5 重测参数。PDB、Packmol、packing box 和 z padding 只保留旧流程兼容能力。
- 重要现实状态：v2 已实现复杂单元和最终相测量框架，但“同时满足全部输入分布”仍不成熟。2026年8月28日的三个可视化示例均生成了 GLB，但严格审核均失败，因此这些 GLB 只能用于观察形状，不能作为合格科学样本。

研究参数口径参考 [飞书参考文档 DL33diGBooDWj8xlyKxcX9Mznzg] 和 [飞书参考文档 FrNVdN3VaoGfS7xDRe7c7NSGnTc]。本文件不修改《字节报告0821》。

## 版本身份与冻结规则

| 项目 | 固定值 |
| --- | --- |
| 正式名称 | Porous Film Generator 2.0 — Complex Shapes |
| Git 标签 | `2.0`，不可移动、不可覆盖 |
| 维护分支 | `release/2.0` |
| 提交 | `1c9a10793e96437202482ec44e4263b87ef64882` |
| Python 包版本 | `0.4.0.dev1`。这是历史内部版本号，不应再用于判断 v1/v2/v3。 |
| 基线关系 | 从标签 `1.0` / 历史别名 `original-v0.2.0` 演进而来；与 v3 的相场孔网络路线分开。 |
| 二次开发起点 | 从 `2.0` 新建分支；不得移动标签，不得把修改直接写回标签快照。 |

从 1.0 到 2.0 共涉及 40 个文件，约新增 11033 行、删除 761 行。核心新增文件是 `geometry/complex_shapes.py` 和 `metrics/final_geometry.py`；独立验证器也大幅扩展。

## 需求演变：最初希望的参数、后来确认的参数、v2实际实现

### 最初设计中的参数

2026年8月12日的初版设计把“每个生成单元”作为主要对象。输入大致分为：

- 总体：生成单元体密度 \lambda_{seed}、孔隙率 \Phi、通道单元比例 f_{channel}、通道与紧凑单元平均体积比 R_V。
- 位置：生成单元中心的三维 RDF，或规则点阵加扰动。
- 大小：紧凑孔与通道孔各自的相对生成单元体积分布。
- 方向：一个孔轴与 x 方向的夹角，绕 x 方位角均匀。
- 形状：紧凑孔长宽比、通道长细比、通道曲折度。
- 表面：roughness，初版曾希望它直接代表表面积增量或曲率变化。
- 硬约束：半导体沿 x 贯通、任意 yz 截面不能被切断、可选最小骨架厚度。
- 分子流程：目标盒、冗余盒、孔相材料 PDB、密度或分子数，以及 Packmol 交接。

这一口径的问题是：多个生成单元会重叠、合并、被边界裁剪，生成单元的体积、中心、方向和数量不再等于最终连通孔的实测结果。

### 后来确认的正式研究参数

经过逐项确认后，正式参数仍按“位置—数量、形状、占比”三类组织，但对象改为最终三维孔相：

| 维度 | 正式参数 | 最终物理含义 |
| --- | --- | --- |
| 位置—数量 | `g_xy(r)` | 同一 z 截面中，不同贯通孔中心线之间的周期 xy 距离分布，沿整个厚度汇总。 |
| 形状 | `p_D(D_eq)` | 沿最终通孔中心线的法向截面等效直径分布。 |
| 形状 | `p_ori(theta_xz, theta_xy)` | 同一三维孔轴在 xz 与 xy 平面的成对投影角分布。 |
| 形状 | `p_eta,c` | 最终紧凑孔的主轴尺寸与两个横向尺寸几何平均值之比。 |
| 形状 | `p_eta,ch` | 最终通道中心线弧长除以通道级等效直径。 |
| 形状 | `p_tau` | 最终通道中心线弧长除以端点直线距离。 |
| 形状 | `p_kappa(w)` | 最终法向截面边界曲率的相对波动分布。 |
| 占比 | `Phi_V` | 真实目标盒内最终孔相体积分数。 |

生成单元数量密度、通道标签比例、单元相对体积、roughness 幅度、叶瓣数、控制点和形状种子被降为内部控制量或诊断量。它们可以帮助造孔，但不能直接证明最终孔结构正确。

### v2代码的实际对应关系

| 参数 | 实际输入路径 | 最终测量 | 当前状态 |
| --- | --- | --- | --- |
| 孔隙率 | `formal_targets.proportion.porosity` | `mean(final_phase.pore_mask)` | 已实现并参与通过判定。 |
| 中心距离 | `formal_targets.position_quantity.center_distance_xy.components` | 最终贯通中心线在同一 z 层的周期 xy 距离。 | 已实现；样本少或无贯通线时会失败。 |
| 局部等效直径 | `formal_targets.shape.equivalent_diameter_A` | 最终贯通中心线法向截面的面积等效直径。 | 最终审核已实现；生成阶段该目标只直接用于通道单元。 |
| 双平面取向 | `formal_targets.shape.orientation.components` | 最终中心线端到端轴的 `theta_xz` 与 `theta_xy`。 | 两个边缘分布分别审核；尚未审核二者的联合配对关系。 |
| 紧凑孔长宽比 | `formal_targets.shape.compact_aspect_ratio` | 设计要求是最终紧凑孔分类后测量。 | 当前 schema v3 审核未实现该最终测量，实际返回 `compact_eta_result=null`。 |
| 通道长细比 | `formal_targets.shape.channel_aspect_ratio` | `L_arc / D_channel` | 已实现并参与通过判定。 |
| 通道曲折度 | `formal_targets.shape.channel_tortuosity` | `L_arc / L_end` | 已实现并参与通过判定。 |
| 曲率波动 | `formal_targets.shape.curvature_fluctuation` | 最终法向截面边界的 `std(kappa)/abs(mean(kappa))`。 | 审核已实现；生成映射仍是经验函数，尚未标定。 |
| 生成单元密度 | `generation_controls.seed_number_density_A3` | 只报告最终中心/轨迹数量。 | 内部控制，不要求同值。 |
| 单元类型与相对体积 | `generation_controls.channel_fraction_by_count` 等 | 只作生成溯源和诊断。 | 不属于正式最终参数。 |

## 输入文件：单位、结构和完整示例

### 基本规则

- 配置为严格 YAML 映射；未知字段会被拒绝。
- 长度统一用 Å；1 nm = 10 Å，1 μm = 10000 Å。
- 公开取向输入用度；内部旋转计算用弧度。
- 比例和概率无量纲；混合权重必须非负且总和为 1。
- 正式 schema 版本写 schema_version: 3。软件正式版本仍称 2.0，这两个数字不是同一层级。
- 标准几何工作流只要求 target_box_A；不需要 PDB、packing box 或 z padding。

### 顶层配置块

| 配置块 | 角色 | 说明 |
| --- | --- | --- |
| `task` | 运行身份 | `name` 和整数 `random_seed`。 |
| `film` | 物理盒 | `target_box_A{x,y,z}`；兼容字段 `packing_box_A`/`z_padding_A` 仅供旧分子流程。 |
| `formal_targets` | 正式研究目标 | 位置—数量、形状、占比三类目标，必须由最终孔相按同义定义测量。 |
| `generation_controls` | 内部造孔控制 | 种子密度、孔型比例、相对体积和起始 roughness。 |
| `measurement` | 测量协议 | 固定采样间距、中心追踪阈值、截面重采样和曲率平滑尺度；同一研究批次应固定。 |
| `matrix_constraints` | 半导体硬约束 | x 贯通、最小 yz 截面、可选骨架厚度和重叠上限。 |
| `audit` | 候选与网格 | 候选数、轮数、粗/细体素间距和内存上限。 |
| `optimization` | 重复种子 | `seed_panel` 用于估计随机波动；不提供外部优化算法。 |
| `parallel` | CPU 并行 | `auto/seeds/candidates/serial`；每个 worker 固定 1 个数值线程。 |
| `pore_material` | 旧分子流程 | 可选 PDB，加目标密度或精确分子数二选一。 |

### 支持的分布

| family | 常用参数 | 说明 |
| --- | --- | --- |
| `constant` | `value` | 所有样本固定为一个值。 |
| `lognormal` | `sigma/s`，`scale` 或 `mean/mu`，可选 `loc` | 正偏分布。 |
| `gamma` | `alpha/shape/k`，`scale/theta`，可选 `loc` | 正值连续分布。 |
| `weibull` | `shape/k/alpha`，`scale`，可选 `loc` | 实现别名 `weibull_min`。 |
| `truncated_normal` | `mean`、`sigma/s`、`lower`、`upper` | 截断正态；别名 `truncnorm`。 |
| `beta` | `alpha`、`beta`、`lower`、`upper` | 有界分布；取向分量必须使用 Beta。 |
| `mixture` | `components[{weight,family,...}]` | 解析分布线性组合；不支持表格 PDF/CDF/直方图。 |

有限样本不是简单独立随机抽取：程序先按最大余数法给各混合分量分配整数样本数，再在每个分量内做分层分位点抽样，因此同一 seed 可复现，小样本下也尽量保持混合权重。

### 测量协议默认值

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `z_slice_spacing_A` | 1.0 | 沿厚度抽取 xy 截面的间距。 |
| `center_min_separation_A` | 2.0 | 同一截面中两个中心的最小分离尺度。 |
| `center_tracking_max_displacement_A` | 4.0 | 相邻截面中心可连接为同一轨迹的最大 xy 位移。 |
| `center_distance_bin_width_A` | 1.0 | `g_xy(r)` 的距离分箱宽度。 |
| `center_distance_reference_samples` | 4096 | 均匀随机参考的 Sobol 样本数。 |
| `centerline_sample_spacing_A` | 2.0 | 中心线重采样间距。 |
| `cross_section_spacing_A` | 2.0 | 法向截面沿弧长的间距。 |
| `boundary_resample_spacing_A` | 0.5 | 截面边界重采样间距。 |
| `curvature_smoothing_length_A` | 1.0 | 计算曲率前的平滑长度。 |
| `branch_exclusion_length_A` | 2.0 | 分支/合并附近剔除范围。 |
| `surface_exclusion_length_A` | 2.0 | 上下表面附近剔除范围。 |
| `orientation_projection_min_fraction` | 0.05 | 投影过小时把对应角标成不可辨识。 |

> 这些默认值只是软件默认，不是适用于微米级薄膜的推荐物理值。实际值必须与体素间距、最小孔径、膜厚和中心线弯曲尺度协调，并在同一优化批次中固定。

### 可直接运行的 schema v3 示例

**示例数字只演示语法，不代表已标定物理参数**

```yaml
schema_version: 3
task:
  name: through-pore-study
  random_seed: 123
film:
  target_box_A: {x: 10000, y: 10000, z: 1500}
formal_targets:
  position_quantity:
    center_distance_xy:
      components:
        - {kind: exclusion, amplitude: 0.9, center_A: 0, width_A: 300}
        - {kind: peak, amplitude: 0.3, center_A: 900, width_A: 180}
  shape:
    equivalent_diameter_A:
      {family: beta, alpha: 2.5, beta: 3.5, lower: 300, upper: 900}
    orientation:
      model: paired_projected_planes
      components:
        - weight: 1.0
          theta_xz_deg:
            {family: beta, alpha: 4, beta: 2, lower: 55, upper: 88}
          theta_xy_deg:
            {family: beta, alpha: 2, beta: 5, lower: 0, upper: 30}
    compact_aspect_ratio:
      {family: beta, alpha: 2, beta: 3, lower: 1, upper: 3}
    channel_aspect_ratio:
      {family: beta, alpha: 2, beta: 2, lower: 3, upper: 12}
    channel_tortuosity:
      {family: beta, alpha: 2, beta: 3, lower: 1, upper: 1.8}
    curvature_fluctuation:
      {family: beta, alpha: 2, beta: 5, lower: 0, upper: 0.8}
  proportion:
    porosity: 0.15
generation_controls:
  seed_number_density_A3: 8.0e-10
  channel_fraction_by_count: 1.0
  channel_to_compact_mean_volume_ratio: 1.0
measurement:
  z_slice_spacing_A: 25
  center_min_separation_A: 50
  center_tracking_max_displacement_A: 75
  center_distance_bin_width_A: 50
  center_distance_max_A: 5000
  center_distance_reference_samples: 8192
  centerline_sample_spacing_A: 25
  cross_section_spacing_A: 25
  boundary_resample_spacing_A: 10
  curvature_smoothing_length_A: 25
  branch_exclusion_length_A: 50
  surface_exclusion_length_A: 50
  orientation_projection_min_fraction: 0.05
matrix_constraints:
  enabled: true
  require_x_percolation: true
  minimum_cross_section_fraction: 0.10
  maximum_overlap_fraction: 0.50
audit:
  enabled: true
  candidate_count_per_round: 8
  maximum_rounds: 4
  coarse_spacing_A: 50
  fine_spacing_A: 25
parallel:
  enabled: true
  strategy: candidates
  max_workers: 8
  cpu_fraction: 0.8
  memory_fraction: 0.75
  worker_threads: 1
  start_method: spawn
```

## 孔结构的生成顺序

1. 解析并标准化配置。Pydantic 使用 extra=forbid 拒绝未知字段。旧 schema v2 可通过 translate_schema_contract() 转成内部 schema v3 视图。
2. 预检查。检查目标盒、PDB（若提供）、种子数、候选数、粗细网格是否整除盒尺寸、预计体素数和内存。
3. 确定生成单元数。N=round(seed_number_density_A3 × Lx × Ly × Lz)。N 小于 1 直接失败，N 小于 5 给出分布样本不足警告。
4. 生成内部锚点。RDF 模式先在盒中随机放点，再对单点做退火式随机移动；x/y 周期回卷，z 截断在上下表面。schema v3 使用绝对 xy 距离目标；每点约 24 次尝试。也保留 lattice_jitter 兼容模式。
5. 分配孔型。按 channel_fraction_by_count 用最大余数法确定紧凑孔与通道孔的整数数量，再随机打乱标签。
6. 抽取尺寸和形状参数。各分布采用分层分位点抽样；通道和紧凑孔相对体积先按类内比例分配。schema v3 还抽取通道绝对等效直径、成对取向和曲率波动目标。
7. 构造紧凑孔。生成多叶局部轮廓，放到锚点并旋转到抽样方向。
8. 构造通道孔。生成三维多弯中心线和变半径曲线，放到锚点并旋转。
9. 组成全局孔场。每个单元提供有符号距离；所有单元用 smooth-min 合并。x/y 使用周期像，z 不周期。
10. 统一缩放求孔隙率。在 0.1—10 的线性尺度区间内二分，最多 64 次；每次重新体素化，直到最终孔隙率进入网格容差。所有半径、叶瓣偏移、中心线相对锚点长度和半径曲线一起缩放，锚点位置不变。
11. 从最终体素孔相重建中心线。逐 z 截面做周期距离变换，寻找局部最大内切距离点，再用相邻截面最小代价匹配追踪中心线；标记分支邻域和上下表面接触。
12. 测量并审核。只用最终孔相计算孔隙率、g_xy(r)、法向截面直径、曲率、方向、通道 eta/tau、连通性、截面和局部厚度稳定性。
13. 选择候选并重放。按固定顺序选择第一个全部通过的候选；若没有通过项，则选择“孔隙率最接近”的成功候选作为明确的不合格诊断结果。随后按同一候选身份重新生成，检查重放一致性。
14. 导出。写出最终相位、中心线、截面、GLB/PLY、生成单元溯源、报告、哈希以及可选的优化交换文件。

## 复杂孔形状算法

### 多叶紧凑孔

- 每个孔使用独立 shape_seed；最多尝试 16 次，失败时整个候选失败，不退回规则椭球。
- 叶瓣数为 2、3、4，概率分别为 0.25、0.50、0.25。
- 叶瓣相对尺度在 0.65—1.15；新叶瓣与父叶瓣中心距是两者标量半径和的 0.58—0.82，主动形成连通腰部。
- 按体积权重归零重心，PCA 对齐主轴，再做仿射变换，使横向包络相同并使主轴/横轴比等于抽样 eta。
- 叶瓣间使用尺度相关 smooth-min；平滑长度为最小横向叶瓣半径的 0.12。
- 固定、无扰动 Sobol 2^15 点估算体积，再统一缩放到目标体积。
- 要求叶瓣连通、eta 误差不超过 5%、包络填充率 0.50—0.85。

### 三维多弯、变截面通道

- 使用 7 个控制点；局部主轴为 x，横向 y/z 由 1—3 阶正弦模式叠加，端点回到主轴。
- 通过根求解横向振幅，使实际弧长/端距符合 tau。
- tau 大于 1.05 时要求至少两个弯折且中心线非平面。
- 沿归一化弧长生成 7 个半径结点，使用 PCHIP 插值。
- 半径变异系数限制为 0.15—0.30；局部半径约束在等效半径的 0.60—1.45；必须同时有鼓包与收颈。
- 非相邻中心线段不得发生管体自交；最多重采样 16 次。
- SDF 使用共享角平分面拼接相邻段，只在全局两端加半球端帽。
- 每段缓存累计弧长、端点半径、最大半径、局部坐标系和扩展 AABB；查询时先用 AABB 下界裁剪，减少完整距离计算。

### roughness

roughness 在宏观多叶或通道轮廓形成后只施加一次。当前实现按 `unit_id` 派生四个正弦模式的幅度与相位。schema v3 的曲率目标先用经验映射 `roughness=min(0.25, 0.05 × target_curvature_fluctuation)` 转成起始扰动；最终仍以实测曲率波动判定。该映射不是已标定的反函数，是二次开发重点。

## 最终结构怎样测量与审核

### 测量器只读最终相位

`metrics/final_geometry.py` 的输入只有 `PhaseGrid` 和固定的 `MeasurementSpec`。它不读取目标分布、生成标签、抽样值、shape_seed 或生成器写出的单孔指标。这样可以避免“拿输入证明自己正确”。

### 主要测量定义

- 孔隙率：孔体素数除以全部目标盒体素数。
- 截面中心：在每个抽样 z 层对孔相做周期距离变换，寻找局部距离峰，并按 center_min_separation_A 去重。
- 中心线：用周期 xy 距离和匈牙利匹配连接相邻层中心；同时接触最下与最上采样层的轨迹标为 through。
- 中心距离：仅在同一 z 层、不同 through 轨迹之间计算周期最短 xy 距离；用相同对数的 Sobol 均匀参考归一化。
- 法向截面：沿平滑后的中心线按物理弧长取点，在局部切向的法平面上三线性采样孔相，选择包含中心点的连通截面。
- 等效直径：D_{eq}=2\sqrt{A_\perp/\pi}。
- 曲率波动：闭合边界按固定间距重采样并平滑后，计算 std(\kappa)/|mean(\kappa)|。
- 取向：用中心线首末点向量；\theta_{xz}=atan2(|n_z|,|n_x|)，\theta_{xy}=atan2(|n_y|,|n_x|)。
- 通道 eta：L_{arc}/D_{channel}，其中 D_{channel}=2\sqrt{mean(A_\perp)/\pi}。
- 通道 tau：L_{arc}/L_{end}。

### 比较方法和当前阈值

分布比较使用两个互补指标。KS 距离看累计分布最大差异；归一化 Wasserstein 距离看把一条分布搬到另一条分布所需的平均距离，再除以目标分布尺度。两者都越小越好。

| 检查 | 当前阈值 | 代码位置 |
| --- | --- | --- |
| 一般一维分布 KS | ≤ 0.20 | `metrics/audit.py::_DISTRIBUTION_KS_LIMIT` |
| 归一化 Wasserstein | ≤ 0.20 | `metrics/audit.py::_DISTRIBUTION_WASSERSTEIN_LIMIT` |
| `g_xy(r)` 加权均方损失 | ≤ 1.0 | `metrics/audit.py::_RDF_LOSS_LIMIT` |
| 孔隙率绝对误差 | ≤ max(1/Nvox, 0.01) | `metrics/audit.py::_porosity_tolerance` |
| 常数通道 eta/tau 相对误差 | 单样本各自 ≤ 1% | 特殊 constant 逻辑；同时仍要求 Wasserstein ≤ 0.20。 |
| 复杂紧凑单元体积 | 独立误差 ≤ 3% | validator 对 unit schema v2 的检查。 |
| 复杂通道单元体积 | 独立误差 ≤ 3% | validator 对 unit schema v2 的检查。 |

结构约束还包括半导体 x 方向贯通、最小 yz 截面比例、可选最小骨架厚度、孔域数量、最大孔域比例、表面开口和局部厚度粗细网格稳定性。

### 独立验证器

`porous_film_validator` 不导入主包 `porous_film`。它重新读取 HDF5/CSV/JSONL/GLB，复算最终相位、中心线、截面、正式分布、连通性、网格与 GLB 占据一致性、单元几何和可选分子信息，并校验 `checksums.sha256`。输出状态为 PASS、FAIL 或 NOT_EVALUABLE。

> 独立实现并不自动等于独立正确。主测量器与 validator 中存在大量镜像代码；外部审查应重点检查两者是否在相同错误假设下同步复制。

## 候选搜索、并行与可复现性

- 候选总数 = candidate_count_per_round × maximum_rounds。
- 父进程从任务 seed 派生每个候选 seed；候选 worker 使用 NumPy RNG，数值库线程固定为 1。
- parallel.strategy 支持 auto、seeds、candidates、serial。
- 并行仅跨候选或 seed；单个候选内部的体素/SDF/截面计算仍以 CPU 为主，GPU 不会自动加速。
- 选择规则是按确定性顺序取第一个通过候选；若无候选通过，则从成功生成者中选孔隙率最接近目标的一项，并保留为不合格诊断。
- 选中候选会在父流程中按同一 identity 重放，并比较孔隙率、尺度、审核状态和警告，以防 worker 与发布结果漂移。
- optimization.seed_panel 可对同一设计点运行多个固定 seed，并汇总可行率、目标均值和方差。

## 安装与使用

### 环境要求

- Python ≥ 3.12；本次本机复核实际使用 CPython 3.14.6。
- 推荐使用 uv 和仓库中的 uv.lock。
- 主要依赖：NumPy、SciPy、scikit-image、h5py、trimesh、Pydantic、PyYAML、Typer、psutil、threadpoolctl、gemmi。
- 生成器本体不需要 GPU；多核 CPU 和足够内存更重要。

### 从源码运行

```powershell
cd porous-film-generator-2.0
uv sync --frozen --all-groups
uv run porous-film version
uv run porous-film preflight --config .\examples\config.yaml --result-root C:\Calculation_results
uv run porous-film generate-geometry --config .\examples\config.yaml --result-root C:\Calculation_results --workers 8
uv run porous-film-validate C:\Calculation_results\YYYY-MM-DD\python_results\TASK\qa_export
```

### 从 wheel 安装

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install porous_film_generator-0.4.0.dev1-py3-none-any.whl
porous-film version
porous-film --help
```

### 命令语义

- preflight：只做配置、网格、PDB、内存和样本量检查，并写预检查报告。
- generate-geometry：生成与审核几何，写主要几何和 QA 产物；当前实现不会像完整 generate 一样保证写出全部优化交换 JSON。
- generate：完整流程。若没有 pore_material，仍可生成几何和完整状态/优化交换；若有 PDB，再执行孔内刚体分子放置和 Packmol 交接。
- fill-pore：当前只检查既有孔填充产物并写状态，不会重新执行通用填充算法。
- audit：汇总既有运行的可行性状态；不会重新从头测量所有几何。
- audit-packmol-output：检查后续 Packmol 结果中非孔相原子/分子质心是否深入原孔域。
- porous-film-validate：对 qa_export 执行独立验证。

### 完成判定

1. 先看命令退出状态，再看 outputs/calculation_status.json。
2. 区分 completed_feasible、completed_infeasible 和 failed。
3. 读取 outputs/feasibility.json 与 realized_geometry_parameters.json。
4. 确认 qa_export/checksums.sha256 完整。
5. 单独执行 validator，并以其 PASS/FAIL/NOT_EVALUABLE 为独立证据。
6. GLB 能打开只证明网格可查看，不证明分布与硬约束合格。

## 输出文件与下游用途

| 文件 | 用途 |
| --- | --- |
| `qa_export/final_phase.h5` | 权威最终二值孔相；数据顺序 z,y,x，0=半导体，1=孔，x/y周期、z有限。 |
| `qa_export/final_centerlines.h5` | 从最终相位提取的中心线、周期展开坐标、壁面距离、贯通和分支标记。 |
| `qa_export/final_cross_sections.csv` | 每个法向截面的中心、切向、面积、等效直径、曲率波动、有效性和剔除原因。 |
| `qa_export/final_measurements.json` | 最终相位测量总表。 |
| `qa_export/unit_geometry.jsonl` | 生成单元 schema v1/v2 溯源；不是正式最终参数证据。 |
| `qa_export/channel_curves.h5` | 生成通道中心线和半径曲线，用于重放与单元级验证。 |
| `qa_export/main_unit_metrics.csv` | 叶瓣数、填充率、半径 CV、弯折、自交净距等生成复杂度诊断。 |
| `outputs/semiconductor_solid_target.glb` | Blender 可打开的目标盒半导体实体，孔表现为空腔。 |
| `qa_export/final_surface.ply` | 中立表面网格。 |
| `outputs/pore_geometry.h5` | 组合几何与生成溯源容器。 |
| `requested_design_parameters.json` | 外部优化器提出的设计点。 |
| `realized_geometry_parameters.json` | 最终实测值、审核结果和形状复杂度摘要。 |
| `feasibility.json` | 可行性与约束状态。 |
| `calculation_status.json` | 运行终态、警告和失败原因。 |
| `objectives.json` | 当前只把孔隙率绝对误差作为占位目标；真实迁移率/热导率需下游写入。 |
| `uncertainty.json` | seed panel 的可行率、均值和方差。 |
| `pore_reference_coordinates.cif` 等 | 仅在启用旧 PDB 分子放置流程时生成，用于 Packmol/后续压缩。 |

## 源码结构与审查入口

| 文件/目录 | 职责与重点入口 |
| --- | --- |
| `src/porous_film/config/models.py` | 严格配置模型、schema v2→v3 兼容翻译、默认值、范围检查和 seed 数计算。 |
| `distributions/core.py` | 解析分布、混合分量整数分配、分层抽样、CDF。 |
| `centers/generation.py` | 点阵扰动、RDF 目标函数、随机中心和退火式中心优化。 |
| `geometry/complex_shapes.py` | 多叶紧凑孔、变半径多弯通道、体积估算和自交检查。 |
| `geometry/sdf.py` | CompactUnit/ChannelUnit、SDF、roughness、周期像、构建单元和缓存。 |
| `geometry/scaling.py` | 统一缩放所有复杂几何字段。 |
| `voxel/grid.py` | 分块体素化、PhaseGrid HDF5、二分尺度求孔隙率。 |
| `metrics/final_geometry.py` | 最终相位中心、轨迹、法向截面、直径、曲率、方向和通道指标。 |
| `metrics/audit.py` | 输入—实测比较、统计阈值、连通/截面/局部厚度审核。 |
| `parallel/*.py` | 资源发现、spawn 进程池、候选/seed 任务、确定性重放、失败记录。 |
| `molecules/*.py` | 可选刚体 PDB 模板读取、孔内放置和碰撞检查。 |
| `io/exporters.py` | GLB/PLY、QA 契约、网格回体素化和校验和。 |
| `pipeline.py` | 端到端编排、文件写出、状态和优化交换，是最大集成入口。 |
| `src/porous_film_validator/validate.py` | 不导入主包的独立验证器；约 4400 行，应重点检查重复实现的一致性。 |
| `tests/` | 配置、分布、形状、体素、测量、并行、导出、验证器和端到端测试。 |

## 当前已确认的缺口与代码审查优先级

> 下面是截至 2026年8月31日从标签 2.0 源码直接核对出的实际缺口。外部团队应把它们当作审查入口，而不是把文档中的理想设计误认为已全部实现。

### 优先级 P0：参数语义与通过判定

1. 紧凑孔 eta 未完成最终相审核。compact_aspect_ratio 可输入并用于生成多叶孔，但 schema v3 路径没有从最终连通孔相分类和测量紧凑孔 eta；compact_eta_result 为 null。
2. 双平面取向只比较两个边缘分布。生成时 xz/xy 角按同一混合分量配对，但审核分别比较 theta_xz 与 theta_xy，没有验证联合配对或相关性。
3. 没有独立的“必须存在 z 贯通孔”配置门。程序能识别 through 轨迹并只用它们做正式测量；没有 through 轨迹时通常因无样本失败，但没有单独、清晰、可配置的拓扑硬约束字段。
4. schema v3 未把最大重叠比例加入通过门。重叠比例会计算和输出，但 _audit_final_geometry_targets() 当前只对 x 贯通和最小截面执行矩阵约束；maximum_overlap_fraction 在旧审核路径才直接判定。
5. 绝对等效直径只直接控制通道生成。紧凑孔仍主要由内部体积和 eta 决定，正式直径目标没有对紧凑孔单元建立直接反演映射。
6. 曲率目标到 roughness 的映射只是经验比例。当前函数是 min(0.25, 0.05*w_target)，不能保证最终曲率分布接近输入。

### 优先级 P1：搜索与可行率

1. 内部只对孔隙率做一维全局尺度二分；中心、通道数量、连接、直径、方向、曲折度和曲率没有联合内层优化。
2. 最终孔合并后，各正式分布会同时变化；当前候选搜索主要靠随机 seed，难以在大盒子里同时满足全部严格目标。
3. 没有合格候选时，回退候选只按孔隙率误差排序，而不是按全部正式分布与硬约束的综合距离排序。
4. 最终审核只统计 through 轨迹。通孔数量少时，直径、方向、eta/tau 和曲率样本可能为 0 或极少，统计检验不稳定。

### 优先级 P2：接口、报告与维护

1. Pydantic 生成的 JSON Schema 仍会把内部兼容字段 pores、center_distribution、compact 标成 required，虽然运行时的 before-validator 会由 schema v3 输入自动补出这些字段；机器接口说明与真实解析行为不完全一致。
2. output.root 和 output.write_plots 当前没有实际生产逻辑；代码本身不生成输入—输出分布曲线。
3. generate-geometry 与 generate 的状态/优化交换输出不完全相同；外部调用方不能假定两条命令写出同样文件。
4. 几何主测量器与独立 validator 有大量重复算法，容易出现双边同时复制同一假设或修复不同步。
5. 官方版本号 2.0、输入 schema 3、包版本 0.4.0.dev1 三套数字并存，外部接口必须记录三者，不能只看 porous-film version。
6. 仓库标签 2.0 没有 LICENSE 文件。向第三方转交、再发布或商业使用前，应由项目所有者补充授权范围。

### 优先级 P3：功能边界

- 不支持 Y 型/树枝型生成通道、闭环孔道和孔内拓扑洞。
- 不支持任意表格 PDF/CDF，只支持解析分布及其线性混合。
- 不直接表达直径、方向、曲率等属性之间的联合协方差。
- GLB 的 Shade Smooth 只改变显示；任何网格几何平滑都必须重新做全部几何审核。

## 当前验证证据

本次交接在一个由标签 2.0 导出的干净源码快照上重新执行：

- uv sync --frozen --all-groups：成功。
- uv run porous-film version：输出 porous-film 0.4.0.dev1。
- uv run ruff check .：通过。
- uv build：wheel 和 sdist 均成功生成。
- 完整测试：441 passed，6 failed，5 skipped，用时约 7分59秒。

六个失败中，五个来自 Windows 嵌套 PowerShell 环境无法解析 `Get-FileHash` 的部署构建测试；一个来自进程池故障时“已完成结果集合”的竞态测试。该并行测试单独重跑三次，结果为两次通过、一次失败，说明它具有时序不稳定性。五个 skipped 均为需要显式设置 `POROUS_FILM_RUN_HEAVY=1` 的重型生成/并行测试。

这份结果不应写成“全部测试通过”。外部团队应在 Python 3.12 的目标 Linux 环境重跑全套测试、重型测试和一个真实分辨率的端到端样本。

## 既有示例与性能证据

交接包附带 10 个 3 × 3 × 0.6 μm 的 schema v3 配置，覆盖分散虫状孔、圆钝杆状孔、细分散孔、多尺度孔、聚集多叶孔、不规则网络和双连续倾向等设计。还附带前三个样例的 GLB 和状态 JSON。

| 样例 | 孔隙率 | 总耗时 | 峰值内存 | 审核结果 |
| --- | --- | --- | --- | --- |
| A01 dispersed-worms | 0.2793 | 1895 s | 3436 MiB | FAIL；无 through 轨迹样本。 |
| A02 rounded-rods | 0.3005 | 2216 s | 3613 MiB | FAIL；分布和粗细网格稳定性未通过。 |
| A03 fine-dispersed | 0.2903 | 1321 s | 3920 MiB | FAIL；无 through 轨迹样本。 |

这些样例证明程序能构造和导出复杂孔形状，但也直接证明当前“生成机制到最终正式分布”的映射需要继续优化。

## 二次开发建议与不可破坏的约束

1. 先固定基线。从标签 2.0 新建开发分支，记录提交、输入 hash、Python/依赖版本和测试结果。
2. 先补语义缺口，再提速。第一批建议依次处理：z 贯通硬门；最终紧凑孔分类/eta；联合取向审核；schema v3 重叠门；曲率—roughness 标定；多目标候选排序。
3. 每个正式输入必须有同义最终测量。对象、单位、边界、权重、分段和剔除规则必须一致；若做不到，应把它降为内部控制量。
4. 保持测量隔离。最终测量器不得读取目标值和生成单元记录；validator 不得导入主包或信任主程序预写指标。
5. 先写失败测试。至少覆盖配置、生成映射、最终测量、主审核、独立 validator、串并行一致性、输出 schema 和端到端样本。
6. 性能优化不能改变科学几何。对 SDF 裁剪、向量化、并行、缓存或 GPU 改写，必须与暴力参考逐点比较，并验证串行/并行输出哈希或数值等价。
7. 不要只优化孔隙率。建议把内部候选搜索改为同时考虑 g_xy、直径、方向、eta、tau、曲率、贯通和半导体约束的多目标/约束优化。
8. 区分预览与生产。粗网格只用于形状筛选；生产结果必须做网格收敛、独立验证和足够的随机种子重复。
9. 保留兼容层但停止扩大旧接口。旧 schema v2、PDB 和 padding 可继续读取；新功能应优先围绕 schema v3 的最终相参数契约。

## 推荐代码审查清单

- 输入：未知字段、分布族参数、支持区间、混合权重、单位与 schema 转换是否严格。
- 随机性：所有随机来源是否由任务 seed/候选 identity/shape_seed 可追踪地派生。
- 形状：多叶孔体积、包络、连通；通道 eta/tau、半径 profile、端帽和自交。
- 边界：x/y 周期像是否无重复/缺失，z 是否保持有限且不被错误回卷。
- 缩放：所有长度字段是否同步缩放，体积是否按三次方变化，方向和 eta/tau 是否不变。
- 体素：轴顺序是否始终为 z,y,x；间距是否严格整除盒长；孔/半导体编码是否一致。
- 测量：中心提取、轨迹连接、分支与表面排除、截面闭合、曲率平滑是否对分辨率稳定。
- 审核：所有通过门是否只使用最终测量；缺样本是否明确失败；不合格候选是否不会被标成合格。
- 独立验证：是否真正不依赖主实现，是否会检测被篡改的 HDF5/CSV/JSON/GLB 和校验和。
- 并行：spawn、取消、worker 失败、结果排序、重放和父进程单写是否无竞态。
- 输出：完整运行与几何运行的文件契约是否一致、版本是否清楚、报告是否会误写状态。
- 安全与发布：不得打包账号、密码、私钥、固定服务器入口；补充明确 LICENSE 后再决定外部分发范围。

## 交接包内容与使用顺序

桌面压缩包按下列顺序阅读：

1. 00-START-HERE.md：版本、验证结果、材料目录和安全说明。
2. docs/v2孔生成器说明.md 与 PDF：本飞书文档的离线副本。
3. source/porous-film-generator-2.0/：去除环境专用远程部署资料后的审查源码。
4. dist/：本次从标签 2.0 构建的 wheel 和 sdist。
5. examples/configs/：10 个实际 schema v3 配置。
6. examples/visual-only-failed-audit/：3 个 Blender GLB 和对应状态；仅看形状。
7. examples/synthetic-validator-pass/：测试生成的最小 QA 数据包，用于理解文件契约，不代表真实材料结构。
8. review/：JSON Schema、CLI help、1.0→2.0 核心差异、提交记录、测试日志、已知问题和文件清单。
9. SHA256SUMS.txt：包内文件校验。

为避免泄露内部环境信息，外部包不包含凭据、私钥、明文密码、真实远程提交 JSON，也不包含带固定服务器入口/本机密钥路径的环境专用部署模板和历史部署计划。核心生成、审核、验证、测试和通用使用文档均保留；删减内容记录在 `SECURITY-REDACTIONS.md`。

## 维护结论

v2最值得保留的资产不是某一张 GLB，而是三条工程原则：复杂孔单元可复现生成；正式参数必须从最终孔相同义测量；主程序与独立 validator 分开核验。外部优化应先补齐参数语义和通过门，再改候选搜索与性能。任何速度提升如果改变最终相位、测量定义或随机身份，都必须视为科学行为变更，而不是普通重构。
