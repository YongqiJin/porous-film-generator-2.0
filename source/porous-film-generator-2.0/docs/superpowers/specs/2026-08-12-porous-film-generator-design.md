# 三维多孔薄膜孔结构生成器设计规格

**日期：** 2026-08-12
**状态：** 待用户书面审阅
**源代码目录：** C:\Calculation_assist\porous-film-generator
**结果目录：** C:\Calculation_results\YYYY-MM-DD\python_results\<task-name>\

## 1. 目标

开发一个可复现、可审核、可供贝叶斯优化调用的三维多孔薄膜生成器。程序接收目标薄膜尺寸、初始冗余盒尺寸、有限维孔结构参数和孔相材料单分子 PDB，完成：

1. 生成满足目标统计分布的紧凑孔和曲折通道孔；
2. 允许孔单元重叠、合并和贯通；
3. 保证半导体相沿 x 方向贯通；
4. 在孔域中放置真实孔相材料分子；
5. 输出 PDB、mmCIF、几何场、Blender 可视化实体、审核报告和优化交换文件；
6. 供后续 Packmol 填充半导体、性能计算和贝叶斯优化使用。

本程序不负责力场、拓扑、电荷、Packmol 半导体填充、GROMACS 压缩平衡或输运计算本身。

## 2. 边界和坐标

- x、y 为周期方向。
- z 为开放表面方向。
- 孔相允许合并、贯通和接触上下表面。
- 半导体相必须沿 x 方向三维贯通，不能被完全切断。
- 每个孔生成单元保留稳定 ID，即使最终与其他单元合并。
- 孔相分子的质量中心必须位于目标孔域。
- 孔相原子允许跨 x/y 周期边界，也允许从目标薄膜上下表面伸出。
- 目标薄膜内部的孔相原子和键包络不得进入半导体相。
- 后续压缩中孔相保存绝对参考坐标，不随盒子仿射缩放。

内部单位：长度 Å、面积 Å²、体积 Å³、角度 rad、密度 g/cm³、摩尔质量 g/mol。

## 3. 两个盒子

### 3.1 目标盒

\[
\mathbf L_{\rm target}=(L_x,L_y,L_z^{\rm target}),
\qquad
V_{\rm target}=L_xL_yL_z^{\rm target}.
\]

目标盒用于定义孔形貌、孔参数、最终目标密度、局部厚度和半导体贯通性。

### 3.2 冗余盒

\[
\mathbf L_{\rm packing}=(L_x,L_y,L_z^{\rm packing}),
\qquad
L_z^{\rm packing}\ge L_z^{\rm target}.
\]

第一版只允许 z 方向增加冗余空间。冗余盒只用于低密度初始填充、容纳表面伸出分子和后续压缩，不参与孔参数归一化，也不用于计算目标半导体分子数量。

目标薄膜默认位于冗余盒中央，也可显式设置上下不对称 padding。直接给冗余盒尺寸和给上下 padding 两种方式互斥。

## 4. 孔生成参数

### 4.1 全局变量

\[
\Theta_{\rm global}
=
\{\lambda_{\rm seed},\Phi,f_{\rm channel},R_V\}.
\]

- \(\lambda_{\rm seed}\)：孔生成单元体密度；
- \(\Phi\)：最终孔并集在目标盒中的体积孔隙率；
- \(f_{\rm channel}\)：通道单元数量比例。
- \(R_V=\langle V_{\rm channel}\rangle/\langle V_{\rm compact}\rangle\)：通道与紧凑孔的平均体积比，默认 1，可作为贝叶斯优化变量。

\[
N_{\rm seed}
=
\operatorname{round}
(\lambda_{\rm seed}V_{\rm target}).
\]

### 4.2 参数化混合分布

不支持表格 PDF、CDF 或直方图输入。复杂、多峰和层级分布使用有限个解析分布线性组合：

\[
p(x)=\sum_{k=1}^{K}w_kp_k(x;\boldsymbol\alpha_k),
\qquad
w_k\ge0,\quad \sum_kw_k=1.
\]

基础分布包括常数、对数正态、Gamma、Weibull、截断正态和 Beta。有限孔数采用分层分位点抽样；各混合分量数量用最大余数法分配。

一次固定维度的贝叶斯优化任务中，混合分量数量和分布族固定，只优化权重和连续参数。

紧凑孔和通道孔分别使用类内均值为 1 的相对体积：

\[
v_c=V_c/\langle V_c\rangle,
\qquad
v_{ch}=V_{ch}/\langle V_{ch}\rangle,
\qquad
\langle v_c\rangle=\langle v_{ch}\rangle=1.
\]

实际单元体积写为：

\[
V_c=V_0v_c,
\qquad
V_{ch}=V_0R_Vv_{ch}.
\]

全局尺度 \(V_0\) 由最终孔隙率搜索确定。单元体积按各自 SDF 在目标盒内、应用 x/y 周期裁剪后的本征体积计算，不使用与合并顺序有关的“增量并集体积”。

### 4.3 孔中心分布

三维自然距离尺度：

\[
\ell_\lambda=\lambda_{\rm seed}^{-1/3},
\qquad
\xi=r\lambda_{\rm seed}^{1/3}.
\]

支持两种互斥模式。用于中心密度和 RDF 的中心定义固定为：

- 紧凑孔：基础超椭球的几何中心；
- 通道孔：周期展开坐标中中心线的弧长加权质心，再映射回基础盒。

该定义不随表面扰动和与其他孔的合并而改变。

#### RDF 模式

\[
\widetilde g_{\rm target}(\xi)
=
1+\sum_{k=1}^{K_g}a_kb_k(\xi).
\]

解析项包括短程排斥、偏好距离峰、抑制谷、宽聚集峰和衰减振荡。程序先检查目标函数非负、远距离趋近 1、统计距离可被盒尺寸表示、邻居数量可实现及必要物理条件。

程序整体优化全部中心位置，使实际 RDF、峰位置、峰高度和累计邻居数接近目标，不能逐孔独立抽样 RDF。

#### lattice_jitter 模式

先生成规则点阵，再添加可控位置扰动。该模式描述周期和近周期孔排列。生成后仍统一计算和审核实际 RDF。

多峰 RDF 只表示多个常见距离尺度，不自动等同于规则排列。

## 5. 孔形状

### 5.1 紧凑孔

第一版使用轴对称长超椭球：

\[
\eta_{\rm compact}=a/b\ge1.
\]

目标体积和长宽比决定主轴和横向尺度。超椭球指数为任务级标量，可选择是否加入优化变量。第一版不支持扁平圆盘孔和一般三轴椭球。

### 5.2 曲折通道

通道由三维样条中心线和沿中心线排列的局部超椭球构成。相邻局部超椭球通过平滑 SDF 并集合并，不保留串珠接缝。整条通道只计为一个孔生成单元。

\[
\eta_{\rm channel}=L_{\rm arc}/D_{\rm cross},
\qquad
\tau_c=L_{\rm arc}/L_{\rm end}\ge1.
\]

第一版通道连续、无分支、非闭合。中心线在周期展开坐标中保存。

### 5.3 取向

只优化整体主方向与 x 方向的夹角 \(\theta\)，绕 x 的方位角均匀随机。

- 紧凑孔：使用实际几何最长主轴；
- 通道：使用中心线点集的整体主方向；
- 接近球形且方向不可辨识的孔不纳入实际取向分布审核。

### 5.4 表面复杂度

\[
w=(S-S_0)/S_0\ge0.
\]

\(S_0\) 是扰动前基础表面积，\(S\) 是扰动后表面积。代码名称为 roughness，目标分布为 \(p_{\rm roughness}(w)\)。表面扰动波长范围相对于孔等效直径固定或作为任务级参数输入。

## 6. 几何表示

权威几何表示为有符号距离场：

\[
d(\mathbf x)<0\Rightarrow\mathbf x\in\Omega_{\rm pore},
\qquad
d(\mathbf x)>0\Rightarrow\mathbf x\in\Omega_{\rm semi}.
\]

紧凑孔使用超椭球 SDF；通道使用连续扫掠或自适应局部超椭球的平滑最小值近似。最终孔相为所有生成单元的平滑并集。

x/y 使用周期最小镜像，z 不周期。体素场和表面网格只用于审核、可视化和中立数据导出。

## 7. 生成流程

1. 读取并标准化配置、单位和 PDB；
2. 检查目标盒与冗余盒；
3. 评估参数可实现性、有限样本误差、盒尺寸、内存和计算量；
4. 确定孔总数、孔型数量和混合分量数量；
5. 分层抽取体积、形状、取向、粗糙度和迂曲度；
6. 用 RDF 或 lattice_jitter 生成孔中心；
7. 构造紧凑孔和通道孔 SDF；
8. 统一缩放全部生成单元，通过二分搜索使最终孔隙率接近目标；
9. 重新计算实际孔参数分布；
10. 检查半导体贯通、最小截面和可选骨架厚度；
11. 计算局部厚度和连通形貌；
12. 只对通过几何审核的候选填充孔相分子；
13. 将目标薄膜放入冗余盒并导出；
14. 输出主报告和独立 QA 数据包。

每轮可生成多个候选。先做粗分辨率审核，再对合格候选做细分辨率审核。硬约束失败不得由综合评分抵消。

## 8. 主要孔参数审核

最终验收第一层是判断实际孔结构是否满足输入孔参数。必须比较：

- \(\lambda_{\rm seed}\)；
- \(\widetilde g(\xi)\)；
- \(\Phi\)；
- 两类孔各自的相对体积分布；
- \(p_\theta\)；
- 两类孔各自的 \(p_\eta\)；
- \(p_{\rm roughness}\)；
- \(p_\tau\)；
- \(f_{\rm channel}\)；
- 各混合分量实际比例。

每个单元同时记录：

- latent/input：抽样目标值；
- realized/geometric：实际构造后的几何值。

分布比较使用 KS 和归一化 Wasserstein 距离；RDF 使用加权误差、峰位置、峰高度和累计邻居数。每项关键指标有独立容差；综合评分只用于合格候选排序。

默认容差：

- 孔隙率绝对误差 0.005；
- 种子密度相对误差 0.02；
- 混合权重绝对误差 0.03；
- KS 0.05；
- 归一化 Wasserstein 0.03；
- RDF 加权误差 0.05。

合并后的贯通孔域不被强行解释为单个孔。生成单元分布按保留的单元 ID 审核；合并效应由最终孔隙率、重叠比例、局部厚度和连通性表达。

## 9. 硬约束与附加统计

### 9.1 半导体贯通

半导体相必须沿 x 周期方向形成非零绕行路径。孔相可以在任意方向贯通。

### 9.2 最小截面

\[
f_{S,\min}
=
\min_x\frac{A_S(x)}{L_yL_z^{\rm target}}
\ge f_{\rm cut}.
\]

### 9.3 可选骨架厚度

若设置 \(h_{\min}\)，将半导体相侵蚀 \(h_{\min}/2\)，侵蚀后的核心仍须沿 x 贯通。\(h_{\min}\) 可手动输入，也可根据 matrix_reference_pdb 给出建议；设为 null 时只检查原始贯通。

### 9.4 局部厚度

孔相局部厚度 \(D(\mathbf x)\) 是完全位于孔相并覆盖该点的最大球直径；半导体局部厚度 \(h(\mathbf x)\) 同理。\(p_V(D)\) 和 \(p_S(h)\) 按对应相体积加权，不能用两倍最近界面距离替代。

局部厚度只在目标盒内计算：x/y 采用周期延拓，z=0 和 z=Lz_target 作为目标形貌边界。伸入冗余空间的分子坐标不计入 \(p_V\)、\(p_S\) 或孔隙率。

附加报告包括连通孔域数量、最大孔域体积分数、孔相各方向贯通、上下表面开口、最小截面位置和两级分辨率稳定性。这些量不替代目标孔参数审核。

### 9.5 Blender 三维实体输出

程序必须输出：

~~~text
semiconductor_solid_target.glb
~~~

该文件表示目标盒中的连续半导体实体：

\[
\Omega_{\rm visual}
=
\Omega_{\rm target}
\setminus
\Omega_{\rm pore}.
\]

要求：

- 外形尺寸严格使用目标盒 \(L_x\times L_y\times L_z^{\rm target}\)，不得使用冗余盒尺寸；
- 半导体区域作为三角网格实体；
- 孔相不生成填充物，表现为真实空腔或表面开口；
- 不包含孔相材料分子、半导体分子、冗余 z 空间或伸出目标盒的原子；
- 使用细分辨率 SDF/相场提取外表面和孔壁；
- 法向从半导体实体指向外部或孔腔；
- z 表面开口不得被人工封盖；
- 穿越 x/y 周期边界的孔在两个相对盒面上保留匹配开口，不为获得封闭网格而改变真实几何；
- GLB 中对象名固定为 SEMICONDUCTOR_SOLID_TARGET；
- 附带半透明材质，便于在 Blender 中观察内部孔壁；
- 在 glTF extras 中写入 length_unit=angstrom、target_box_A、periodic_axes、porosity、mesh_resolution_A 和生成任务 ID。

如果孔不穿越盒面，网格应为闭合流形。若孔穿越周期面或开放 z 表面，允许出现与真实边界相对应的开口边，但不得出现非物理裂缝、重复面、自交或孤立坏三角形。

独立验证器通过重新体素化 GLB 网格，将其与 final_phase.h5 中的半导体相比较；对称差超过网格容差时判为失败。

## 10. 孔相分子填充

### 10.1 模板和数量

输入 PDB 作为刚体模板，只允许整体平移和旋转。程序验证元素、坐标、原子数、摩尔质量和文件哈希。

精确数量和目标密度二选一：

\[
N_{\rm mol}
=
\operatorname{round}
\left(
\frac{\rho_{\rm target}V_{\rm pore}N_A}{M_{\rm mol}}
\right).
\]

\(V_{\rm pore}\) 是目标盒内最终孔相体积，不包含伸出目标盒外的空间。

### 10.2 放置规则

- 分子质量中心位于目标盒内孔相；
- x/y 使用周期最小镜像；
- 原子允许跨 x/y 边界；
- 原子允许从目标盒上下开口伸入冗余空间；
- 目标盒内部的原子和键包络不得穿入半导体相；
- 分子间满足最小距离。

填充采用快速随机插入、局部重排和整体平移/旋转优化。目标不可实现时必须区分算法未收敛和几何不可容纳，不得静默减少分子数。

### 10.3 密度审核

审核整体密度、连通孔域局部密度、大空腔、最小原子距离、表面伸出分子数和最大伸出距离。

## 11. Packmol 交付

不生成临时阻挡原子。交付结构只包含真实孔相材料。

输出：

- semiconductor_solid_target.glb；
- pore_material.pdb；
- pore_material_high_precision.cif；
- molecule_instances.csv；
- pore_geometry.h5；
- packing_metrics.json；
- packmol_handoff.inp 模板和盒元数据。

程序不自动执行半导体 Packmol 填充。可选命令 audit-packmol-output 检查后续两相结构中半导体进入原孔域的比例、最大深度和界面混合层。界面附近允许配置的分子级混合，深入孔域的异常团簇判为失败。

## 12. 绝对孔坐标与后续压缩

孔相导出时保存绝对参考坐标。冗余 z 只用于初始低密度填充，不属于孔形貌定义。

输出：

- pore_reference_coordinates.cif；
- phase_mapping.json；
- pore_atom_indices.ndx；
- compression_metadata.json。

后续可交替锁死孔相和半导体相并交替松弛，但属于下游 MD。生成器不自动执行压缩，也不把冻结阶段压力当作最终平衡压力。

后续审核应确认孔相没有缩放或漂移，并重新计算孔参数分布。

## 13. 贝叶斯优化接口

### 13.1 可优化性结论

所有已经参数化且有限维的物理孔生成参数都可作为贝叶斯优化变量。不是所有配置字段都应优化；分辨率、随机种子、审核容差和冗余盒尺寸属于运行条件。

### 13.2 默认可优化变量

- \(\lambda_{\rm seed}\)、\(\Phi\)、\(f_{\rm channel}\)、\(R_V\)；
- RDF 项的幅度、中心和宽度，或 lattice_jitter 的间距和扰动；
- 两类孔相对体积分布的混合权重和参数；
- 取向分布的混合权重和 Beta 参数；
- 紧凑孔长宽比分布；
- 通道细长程度和迂曲度分布；
- 两类孔粗糙度分布；
- 可选的超椭球指数和表面扰动尺度。

### 13.3 建议固定或分任务处理

一次优化任务中建议固定：

- RDF 或 lattice_jitter 模式；
- 点阵类型；
- 混合分量数量；
- 各分量分布族；
- 材料和 PDB；
- 目标盒、边界和冗余盒；
- 孔相填充密度；
- 审核分辨率和容差。

这些离散结构可由混合变量优化器处理，但第一版建议分成独立优化任务。

### 13.4 参数变换

- 正值变量在对数空间优化；
- \([0,1]\) 变量使用有界或 logit 表示；
- 混合权重用 softmax 映射到和为 1 的单纯形；
- 多峰中心使用有序参数化，避免标签交换；
- 只有 \(f_{\rm channel}>0\) 时启用通道条件变量。

### 13.5 随机噪声

同一输入可因随机种子产生不同孔图和性能。一个参数点必须支持多个预设种子，输出性能均值、方差和几何实现误差。推荐固定种子面板，公平比较候选参数。

结构生成失败、分布不合格或半导体不贯通作为约束失败返回，不能伪造性能值。优化器同时学习性能目标和可行域。

### 13.6 标准交换文件

每个候选输出：

- requested_design_parameters.json；
- realized_geometry_parameters.json；
- feasibility.json；
- calculation_status.json；
- objectives.json；
- uncertainty.json。

下游性能计算将迁移率、热导率、比值或多目标指标写入 objectives.json，优化器无需解析日志。

## 14. 软件结构和命令

源代码模块：

~~~text
src/porous_film/
  config/
  distributions/
  centers/
  geometry/
  optimization/
  voxel/
  metrics/
  molecules/
  io/
  reporting/
  cli/
~~~

命令：

~~~text
porous-film preflight --config config.yaml
porous-film generate-geometry --config config.yaml
porous-film fill-pore --run <task-directory>
porous-film generate --config config.yaml
porous-film audit --run <task-directory>
porous-film audit-packmol-output --run <task-directory> --structure <pdb>
~~~

## 15. 结果目录

~~~text
C:\Calculation_results\YYYY-MM-DD\python_results\<task-name>\
  inputs\
  work\
  outputs\
  analysis\
  reports\
  logs\
~~~

禁止静默覆盖。新任务冲突时追加 -02 或时间戳。

主要输出包括 semiconductor_solid_target.glb、PDB、mmCIF、SDF/体素 HDF5、表面 PLY、生成单元记录、目标—实际分布、候选得分、随机种子、版本、哈希和 Markdown 报告。

## 16. 独立验证

独立验证器不导入主程序模块，并在读取主程序指标前完成独立复算。主程序导出版本化中立数据包：

~~~text
qa_export/
  contract.json
  normalized_config.yaml
  unit_candidates.jsonl
  unit_geometry.jsonl
  channel_curves.h5
  final_phase.h5
  final_surface.ply
  main_unit_metrics.csv
  main_metrics.json
  molecules/
    source/
    instances.csv
    placed_atoms.h5
    placed_structure.cif
  checksums.sha256
~~~

contract.json 必须固定长度单位、坐标原点、目标盒、冗余盒、周期条件、网格轴顺序、相编码、各指标定义和格式版本。每个单元记录候选/接受/拒绝状态、稳定 ID、孔型、中心、周期像、抽样参数和实际几何参数。mmCIF/HDF5 是高精度权威坐标；PDB 是 Packmol 交付格式。
独立复算：

- 生成单元数量和混合比例；
- RDF；
- 体积、方向、长宽比、粗糙度和迂曲度；
- 最终孔隙率和连通性；
- \(p_V(D)\)、\(p_S(h)\)；
- 半导体贯通和截面；
- 分子数量、刚体完整性、密度和碰撞。

独立报告同时判断：

1. 主程序报告能否被复现；
2. 独立结果是否满足目标。

状态为 PASS、WARNING、FAIL 或 NOT EVALUABLE。强制数据缺失或关键指标不可评估时按失败处理。实现完成后重新启用已建立的独立验证子 agent。

## 17. 第一版非目标

- 分支和闭合通道；
- 扁平圆盘孔和一般三轴超椭球；
- 表格化概率分布；
- 柔性分子构象变化；
- 自动运行 Packmol 半导体填充；
- 自动执行 GROMACS 压缩和平衡；
- 力场、拓扑和电荷生成；
- 电/热输运计算；
- 贝叶斯优化器本身。

生成器只提供稳定、有限维、可复现的结构设计接口和标准优化交换文件。

## 18. 验收标准

1. 同一配置、随机种子和版本产生相同结果；
2. 支持 RDF 和 lattice_jitter；
3. 支持紧凑孔和无分支曲折通道；
4. 支持参数化混合分布，不支持表格分布；
5. 实际孔参数通过独立容差审核；
6. 半导体沿 x 贯通并满足截面约束；
7. 正确计算 \(p_V(D)\)、\(p_S(h)\) 和连通性；
8. 孔相刚性分子达到目标数量或密度并通过碰撞审核；
9. 支持目标盒和冗余盒，孔绝对参考坐标不变；
10. 输出可被 Packmol、性能计算和贝叶斯优化读取；
11. 独立验证器可复算关键指标；
12. 所有输入、输出、版本、随机种子和失败原因可追溯；
13. semiconductor_solid_target.glb 可由 Blender 导入，尺寸等于目标盒，半导体为实体且孔为真实空腔。
