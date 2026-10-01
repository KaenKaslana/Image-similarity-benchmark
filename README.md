# Image Similarity Benchmark

一个简单、稳定、可解释的 Python 图片相似度 Benchmark。

输入一组 **reference** 图片和一组 **candidate** 图片，按**相同文件名（不区分大小写）**一一配对，
对每一对图片计算 SSIM、LPIPS、Silhouette IoU 和 Edge similarity，输出每张图片的 0–100 分数和所有图片的综合分数。

```text
reference/front.png <-> candidate/front.png
reference/side.png  <-> candidate/side.png
reference/top.png   <-> candidate/top.png
```

不同文件名的图片（例如不同视角）**永远不会**互相比较。

> 核心只比较二维图片，不训练模型。`compare-models` 子命令可以额外把两个三维模型（本地 glb/obj/stl 文件，
> 或 Sketchfab 上可下载的模型）用完全相同的正交相机渲染成六视图（前后左右上下），再交给同一套流程打分；
> 渲染用纯 numpy 实现，不需要 Blender 或 OpenGL。见[三维模型对比](#三维模型对比)。

---

## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [CLI 用法](#cli-用法)
4. [三维模型对比](#三维模型对比)
5. [AI 复刻模型并打分](#ai-复刻模型并打分)
6. [Agent 测评（配置文件驱动，无头运行）](#agent-测评配置文件驱动无头运行)
7. [输入要求](#输入要求)
8. [预处理流程](#预处理流程)
9. [指标说明](#指标说明)
10. [分数计算](#分数计算)
11. [输出文件](#输出文件)
12. [配置文件](#配置文件)
13. [测试](#测试)
14. [重要说明与局限性](#重要说明与局限性)
15. [项目结构](#项目结构)

---

## 安装

环境要求：Windows 11 / Linux / macOS，Python 3.10+（开发使用 3.11），可选 NVIDIA GPU（CUDA）。

### 1. 创建虚拟环境

conda：

```powershell
conda create -n imgsim python=3.11 -y
conda activate imgsim
```

或 venv：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装 PyTorch

有 NVIDIA 显卡时先安装 CUDA 版本（以 CUDA 12.4 为例，其它版本见 <https://pytorch.org/get-started/locally/>）：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

没有显卡时直接安装 CPU 版本：

```powershell
pip install torch torchvision
```

### 3. 安装其余依赖

```powershell
pip install -r requirements.txt
```

### 4. 验证

```powershell
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

程序会自动选择设备：CUDA 可用时使用 GPU，否则使用 CPU（无需修改配置）。

> 首次运行 LPIPS 时，`lpips` 包会自动下载 AlexNet 预训练权重（约 230 MB）到 `~/.cache/torch`。

---

## 快速开始

```powershell
# 1) 生成示例数据到 data/reference 和 data/candidate（可选，用于试运行）
#    默认生成 6 个不同物体 x 三视图 = 18 对图片，见下文「内置测试用例」
python scripts/make_sample_data.py

# 2) 运行 benchmark
python -m src.cli compare --reference data/reference --candidate data/candidate --output outputs --config configs/default.yaml
```

终端输出示例：

```text
name                        SSIM   LPIPS     IoU    Edge    Pair
----------------------------------------------------------------
bottle_front.png           0.977   0.039   0.920    85.9   94.74
bottle_side.png            0.977   0.039   0.920    85.9   94.74
bottle_top.png             0.994   0.006   1.000   100.0   99.56
house_front.png            0.954   0.103   0.957    94.1   93.27
...
table_top.png              1.000   0.000   1.000   100.0  100.00
----------------------------------------------------------------
[bottle] (3 views)                                         96.34
[house] (3 views)                                          92.50
[lamp] (3 views)                                           72.21
[mug] (3 views)                                            96.16
[robot] (3 views)                                         100.00
[table] (3 views)                                          86.51
----------------------------------------------------------------
overall_score                                              90.62

Outputs written to: outputs\run_20260912_135214
```

把你自己的图片放进 `data/reference/` 和 `data/candidate/`（文件名一一对应）后再次运行即可。

### 物体三视图的用法

这是本项目的典型场景：reference 是原始物体的正交三视图（`front.png` / `side.png` / `top.png`），
candidate 是另一个版本（例如重建或重新建模后的物体）用**相同相机设置**渲染出的三视图。

```text
data/reference/front.png  <->  data/candidate/front.png   正视图只和正视图比
data/reference/side.png   <->  data/candidate/side.png    侧视图只和侧视图比
data/reference/top.png    <->  data/candidate/top.png     俯视图只和俯视图比
```

* 每个视角单独得到一个 `pair_score`，`overall_score` 是所有有效配对分数的平均值；不同视角之间**永远不会互相比较**。
* 视角数量不限：加上 `back.png`、`left.png`、`iso.png` 等同名文件即可自动配对。

### 一次比较多个物体

把多个物体放进同一对文件夹时，用 `<物体名>_<视角>.png` 命名：

```text
data/reference/mug_front.png   data/candidate/mug_front.png
data/reference/mug_side.png    data/candidate/mug_side.png
data/reference/mug_top.png     data/candidate/mug_top.png
data/reference/lamp_front.png  data/candidate/lamp_front.png
...
```

程序会按第一个 `_` 之前的前缀把配对分组，在 `metrics.json` 的 `group_scores`、`metrics.csv` 的 `__group__:<物体>` 行
以及终端表格里给出**每个物体的平均分**（该物体三个视角 `pair_score` 的平均）。分隔符可通过
`output.group_separator` 修改，设为空字符串则关闭分组；文件名里没有分隔符的配对不参与分组，只计入 `overall_score`。

### 内置测试用例

`python scripts/make_sample_data.py` 默认生成 6 个**互不相同**的物体，每个物体三视图，candidate 各带一种不同类型的偏差：

| 物体 | candidate 的偏差 | 预期表现 |
| --- | --- | --- |
| `robot` | 完全相同 | 三个视角都 ≈ 100（对照组） |
| `mug` | 少了把手 | front / top 的 IoU 与 Edge 下降；side 视角把手被杯身遮挡，几乎不变 |
| `table` | 桌面变厚、桌腿变短（比例变化） | front / side 明显下降；top 不变 |
| `bottle` | 瓶颈更细更高 | front / side 中等下降；top 只有瓶盖圆略有变化 |
| `house` | 屋顶换色 + 加了烟囱 | 三个视角 LPIPS / SSIM 都下降；轮廓只在 front / side 略降 |
| `lamp` | 同一物体，整体平移 (5 %, 3 %) | 默认对齐后 IoU 0.97、物体分 95.8，位置偏差不扣分；`--alignment none` 时降到 72，说明对齐的作用 |
| `mismatch` | reference 是椅子，candidate 是台灯（两个完全不同的物体） | front / side 的 IoU 只有 0.3 左右，Edge、LPIPS 都很差，是分数最低的一组（69）；但 top 视角两者都是居中的一团（椅面 vs 灯罩），IoU 仍有 0.75，说明单看俯视图分不清物体，多视角一起看才可靠 |

报告按物体组织：**每个物体一张 `report_<物体>.png`**，三行分别是 front / side / top 的 reference、candidate、差异图和指标，
标题是该物体的分数和各视角分数；`report.png` 是一页汇总表（物体 × 视角分数、物体分数、总分），不再重复堆放所有图片。
只有一个物体、文件名不带前缀（`front.png` / `side.png` / `top.png`）时，`report.png` 就是这个物体的三视图报告。

这组用例展示了每个指标分别对什么敏感，也说明了「同一视角、同一位置」这个前提有多重要。物体定义在
[src/objects.py](src/objects.py)，用 box / cylinder / sphere / prism / cone 等基本体拼成并做正交投影，
可以照着添加自己的物体或偏差。其他两组示例：`--kind three-view`（单个长方体 + 圆柱）和 `--kind shapes`（抽象形状）。

生成三视图时的建议：

1. 两套图片使用**相同的相机类型（正交/透视）、相同的视角、相同的距离/缩放、相同的画幅**。
2. 尽量导出带 alpha 通道的 PNG（透明背景），这样 Silhouette IoU 可以直接用 alpha 得到可靠的轮廓；
   否则请用纯色背景，程序会用背景色阈值分割前景。
3. 物体在画面里的**位置**不一致没关系：默认的 `alignment: phase_correlation` 会先把 candidate 平移对齐再打分。
   如果**大小**也不一致（取景框远近不同），再加 `--crop-mode foreground_bbox`，按轮廓包围盒裁剪并居中；
   但注意 bbox 裁剪按轮廓外延对齐，candidate 少了或多了零件时会把整个物体带偏，只在轮廓范围可信时使用。
   如果位置本身就是你要评价的内容，用 `--alignment none`。
4. 光照、材质、颜色的差异会明显影响 SSIM 和 LPIPS；如果只关心几何，请适当提高 `silhouette` 和 `edge` 的权重。

---

## CLI 用法

### 比较两个文件夹

PowerShell 多行：

```powershell
python -m src.cli compare `
  --reference data/reference `
  --candidate data/candidate `
  --output outputs `
  --config configs/default.yaml
```

单行（PowerShell / cmd / bash 通用）：

```powershell
python -m src.cli compare --reference data/reference --candidate data/candidate --output outputs --config configs/default.yaml
```

### 比较单独一对图片

PowerShell 多行：

```powershell
python -m src.cli compare-pair `
  --reference data/reference/front.png `
  --candidate data/candidate/front.png `
  --output outputs
```

单行：

```powershell
python -m src.cli compare-pair --reference data/reference/front.png --candidate data/candidate/front.png --output outputs
```

### 常用选项

| 选项 | 说明 |
| --- | --- |
| `--config PATH` | YAML 配置文件。默认使用 `configs/default.yaml`（存在时），否则使用内置默认值 |
| `--output DIR` | 输出根目录，每次运行在其中创建 `run_时间/`。默认 `outputs/` |
| `--skip-unmatched` | 仅 `compare`：跳过没有配对的文件，而不是终止运行 |
| `--crop-mode {none,center_crop,foreground_bbox}` | 覆盖配置中的裁剪模式 |
| `--alignment {none,centroid,phase_correlation}` | 覆盖对齐方式（默认 `phase_correlation`，位置偏差不扣分） |
| `--canvas-size N` | 覆盖画布尺寸（默认 512） |
| `--device {auto,cuda,cpu}` | 覆盖 LPIPS 计算设备 |
| `--no-save` | 不写任何文件，仅在终端输出结果 |
| `--log-level LEVEL` | DEBUG / INFO / WARNING / ERROR |

退出码：`0` 成功；`1` 没有任何有效配对；`2` 配置错误；`3` 输入/配对错误。

---

## 三维模型对比

`compare-models` 把「下载模型 → 渲染六视图 → 配对打分」串成一条命令：

```powershell
# 两个本地模型
python -m src.cli compare-models --reference models/a.glb --candidate models/b.glb

# 一个本地模型 vs 一个 Sketchfab 模型（需要 API token，见下文）
python -m src.cli compare-models --reference models/a.glb --candidate https://sketchfab.com/3d-models/coffee-mug-<uid>

# 只渲染六视图，不打分
python -m src.cli render-views --model models/a.glb --output renders/a            # 默认六视图
python -m src.cli render-views --model models/a.glb --output renders/a --views front,side,top

# 只下载 Sketchfab 模型（缓存到 models/<uid>.glb）
python -m src.cli fetch-sketchfab https://sketchfab.com/3d-models/coffee-mug-<uid>
```

输出目录名会带上比较对象，例如 `outputs/run_20260915_101010_victorian-chair_vs_tripo-text-v3.1/`
（Sketchfab 模型用其名称，AI 生成的模型用 `服务-方式-版本`，本地文件用文件名；`--label` 可自定义）。目录里会多出：

* `renders/reference/` 与 `renders/candidate/`：渲染出的 `front.png` / `back.png` / `side.png` / `left.png` / `top.png` / `bottom.png`（RGBA、透明背景）及 `views.json`（渲染参数与网格统计）；
* `models.json`：两个模型的来源（本地路径或 Sketchfab 元数据：名称、作者、许可证）。

其余文件（`metrics.json`、`report.png` 等）与 `compare` 完全相同，另外多出面数相关的三项（见下节）。

### 面数（mesh_complexity）

图片指标只看轮廓和明暗，200 个面的方块杯子和 20 万面的精细杯子只要外形一致分数就一样。所以 `compare-models` / `reproduce`
会额外比较两个网格的面数（trimesh 加载后的三角面数，多部件已合并；参考模型如果是四边面，同样按三角化后的数量算）：

```text
shape_score                                                92.86        # 纯图片分，和以前的 overall_score 含义相同
mesh_score               faces 2,304 vs 7,164 (x3.109)     84.08   weight 0.15
mesh_bonus               fewer faces: +0.0 % of 5 %
overall_score                                              90.64
```

面数越少越好（`configs/shape.yaml` 的 `mode: fewer_is_better`）：形状分已经说明像不像，同样像的模型里更省面的赢。

* 面数**多于**参考：`mesh_score`（0–100）由 `log2(候选面数 / 参考面数)` 决定，不到 2 倍得 100，之后线性下降，32 倍得 0；
  `overall = shape_score × (1 − weight × (1 − mesh_score / 100))`，最多扣 `weight`（默认 0.15，即 15%）。
* 面数**少于**参考：不扣分，反而有效率奖励 `bonus`（0–1，少到参考的四分之一时拉满，`bonus_log2: 2`），
  `overall` 再乘 `(1 + bonus_weight × bonus)`，最多加 `bonus_weight`（默认 0.05，即 5%），总分封顶 100。
  奖励也是乘法，所以形状很差的模型省面也救不回来（3 分变 3.15 分）。
* 例子：gpt-6 做的马克杯 2,298 面对参考 2,304 面，不加不减；文字版马克杯 7,164 面（3.1 倍），扣 2.2 分；
  一把 72 面的椅子对 96 面的参考，加 1%。
* `mode: symmetric` 是旧行为：面数太少（欠建模）和太多（AI 扫描式稠密网格）受同样的惩罚，没有奖励。
* 用乘法而不是加权平均，是为了不破坏已校准的 shape 分数尺度（相同 100、无关约 3）。
* `metrics.json` 多出 `shape_score` 和 `mesh`（两边的面数、顶点数、`face_ratio`、`mode`、`score`、`weight`、`bonus`、`bonus_weight`），
  `metrics.csv` 多出 `__shape__` 和 `__mesh__` 两行，`report.png` 标题多一行面数信息，`outputs/results.md` 多 `shape` / `faces` / `mesh` 三列。
* `--mesh-weight 0` 只报告面数不计分（老的 `overall_score` 行为）；也可以在 YAML 里改 `mesh_complexity.mode` / `weight` /
  `free_log2` / `zero_log2` / `bonus_weight` / `bonus_log2`。纯图片的 `compare` 命令不受影响。

### 骨骼与动画（rig）

人物、龙、机甲这类会动的参考模型通常带骨骼（glTF `skins`）、顶点权重和动画片段。渲染只看静止姿态，所以 `compare-models` /
`reproduce` 会直接解析两个 glb/gltf 的结构，在**参考模型有骨骼时**给候选模型打一个 `rig_score`（0–100）：

```text
rig_score                bone 53 skel 53 skin 100 anim 100 moti 83   77.99   weight 0.20
  reference              bones=58 clips=1 motion=0.067
  candidate              bones=11 clips=1 motion=0.024
```

| 分项 | 权重 | 含义 |
| --- | --- | --- |
| `bones` | 0.15 | 骨骼数量比（候选/参考）按 log2 打分：2 倍以内 100，16 倍归 0 |
| `skeleton` | 0.25 | 骨骼位置：两副骨架放进和渲染相同的坐标系（同样的 up/front/yaw、按包围盒归一化）后，关节点集的 Chamfer 距离；0 得 100，`skeleton_max_distance`（默认 0.25 个模型尺寸）归 0 |
| `skinning` | 0.15 | 候选模型蒙皮顶点里权重非零且归一化的比例 |
| `animation` | 0.25 | 候选模型动画覆盖的骨骼比例相对参考的比例（参考没有动画时此项和下一项不计） |
| `motion` | 0.20 | 用线性混合蒙皮把候选模型摆到最长片段的 8 个姿态，算平均顶点位移（除以模型尺寸），和参考的比值按 log2 打分；姿态爆炸或出现 NaN 时动画两项记 0 |

* 没有骨骼的候选模型 `rig_score` 为 0；参考模型本身没骨骼（杯子）时只报告候选的情况、不计分。
* 合成方式同面数：`overall = shape_score × (1 − weight × (1 − rig_score/100))`，`configs/shape.yaml` 默认 weight 0.20，最多扣 20%。
  `--rig-weight 0` 只报告不计分。
* `python -m src.cli rig-info --model x.glb` 单独打印一个模型的骨骼 / 蒙皮 / 动画信息（JSON）。
* `metrics.json` 的 `rig` 里有 `applicable`、`score`、各分项 `components`、`skeleton_chamfer` 和两边的完整信息；
  `metrics.csv` 多 `__rig__` 行；`results.md` 多 `rig` 列。
* 只支持 glTF / GLB（obj、stl 等没有骨骼信息）；不支持 sparse accessor 和 morph target（形态键动画不算动画）。
  动画只比"动了多少、动了多少骨头"，不比动作内容：参考是攻击动画、候选是走路，只要幅度和覆盖相近就得高分。

### 渲染方式

* 支持 trimesh 能读取的格式：glb / gltf / obj / stl / ply / off / 3mf / dae 等；**不支持 fbx**。
  场景中的多个部件会合并成一个网格（节点变换已应用）。
* 模型先按 `--up` / `--front` 旋转到标准姿态（+Y 向上、+Z 朝向正视图的观察者），
  再按包围盒居中并把**最大边长**缩放到 1。三个视图共用同一个比例，所以各视图的相对尺寸保持一致。
* 正交投影，无透视。视图遵循第三角投影法：`front` 从 +Z 看，`side` 从 +X 看（模型正面在图像左侧），
  `top` 从 +Y 看（模型正面在图像底部），`back` / `left` / `bottom` 是对面的三个。默认渲染全部六个；`--views front,side,top` 可退回三视图。
* 着色为平面 headlight：灰度 = 环境光 + 面法线与视线夹角，两个模型使用完全相同的光照。`--style silhouette` 输出纯黑剪影。
* 默认 2 倍超采样抗锯齿（`--supersample`），画布 512 px（`--size`），物体最大边占画布 85 %（`--fill`）。
* 渲染是纯 numpy 的 z-buffer 光栅化，8 万面的网格单个视图约 0.5 s；不需要显卡、OpenGL 或 Blender。

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--views` | `all`（front, back, side, left, top, bottom） | 逗号分隔的视图名；`iso` 需显式指定 |
| `--up` | `+y` | 模型的向上轴。glTF 规范是 +Y；Blender / 很多 STL 是 +Z |
| `--front` | 随 `--up` | 模型正面朝向的轴（+Y 向上时默认 +Z，+Z 向上时默认 -Y） |
| `--size` | 512 | 渲染分辨率 |
| `--fill` | 0.85 | 最大边占画布的比例 |
| `--style` | shaded | `shaded` 或 `silhouette` |

**朝向是最容易出错的地方**：两个模型如果「正面」定义不一致（一个 +Z 朝前、一个 -Y 朝前），即使模型一样分数也会很低。
三种处理方式：

* `--auto-orient`：先枚举 candidate 的全部 24 种轴对齐朝向，渲染低分辨率剪影，选与 reference 各视图 IoU 平均值最高的那一种；
  再绕选出的向上轴每 10° 扫一圈、在最佳角度附近按 2° 细化（AI 生成的模型通常向上轴是对的，但朝向跟着输入图片的拍摄角度走）。
  选中的 `up` / `front` / `yaw` 和前几名的 IoU 写在 `models.json` 的 `candidate.auto_orient` 里。不处理绕其他轴的倾斜和镜像。
  比较剪影时每个视图各自缩放到铺满画布（和 `shape.yaml` 打分时按轮廓裁剪一致），否则搜索会"用旋转换尺寸"：
  模型按最长边归一化，候选的把手长一点整个杯子就显得小一圈，把把手转斜能缩小包围盒、把杯身放大回来——
  实测一个马克杯复刻因此被转了 26°，得 80.3 分，修正后选 0°、84.1 分。剪影不需要深度和着色，
  小三角形走向量化光栅、大三角形用 OpenCV 填充，超过 6 万面的网格先按体素网格合并顶点再搜索（只用于选角度），
  AI 生成的百万面模型对齐从约 2 分钟降到 20–30 秒。
  剪影搜索只负责提名：它选了非 0 的 yaw 时，会把 0° 的轴对齐朝向也完整打一遍分，取综合分高的那个
  （`models.json` 的 `auto_orient.scored` 记录两者的分数）。光看剪影分不清"把手粗了一点、没转"和"把手对、转了 20°"，
  但综合分（边缘、最弱视图）分得清：实测一个文生马克杯剪影搜索选 20° 得 82.8 分，按综合分改选 0° 得 90.6 分。
  代价是多一次打分（约 35 秒），只在选到非 0 角度时发生。
  实测：把一把椅子绕 X、Z 各转 90° 后直接比较只有 63.8 分，加 `--auto-orient` 后恢复到 100 分；
  Tripo v3.1 按 45° 视角图生成的椅子，只做 90° 对齐是 70.2 分，加上任意角度搜索（找到 -42°）后 91.3 分。
* `--candidate-up` / `--candidate-front` / `--candidate-yaw`：手动为 candidate 指定与 reference 不同的轴和绕向上轴的角度。
* 先用 `render-views` 分别看一眼各视图，再决定参数。

### 从 Sketchfab 读取模型

1. 登录 Sketchfab，在 **Settings → Password & API** 复制 **API token**。
2. 设置环境变量（或每次传 `--token`）：

   ```powershell
   $env:SKETCHFAB_API_TOKEN = "xxxxxxxx"        # PowerShell
   export SKETCHFAB_API_TOKEN=xxxxxxxx           # bash / zsh
   ```

3. 用模型页面的 URL、`sketchfab:<uid>` 或 32 位 uid 作为 `--reference` / `--candidate` / `--model` 的值。
   `skfb.ly` 短链接不支持，请用完整 URL。

说明与限制：

* 只有作者开启了 **Download** 的模型（通常是 CC 许可）才能通过 API 下载；未开启的模型会报错并给出原因。
  下载的模型会把名称、作者、许可证写进 `models/<uid>.json`，使用时请遵守相应许可证。
* 下载 API 返回 glb（优先）或 gltf 压缩包；压缩包会自动解压并定位到 `scene.gltf`。
* 模型缓存在 `--models-dir`（默认 `models/`），重复使用不再联网；`fetch-sketchfab --force` 可强制重新下载。
* 认证使用 API token（`Authorization: Token ...`）；也可通过 `SKETCHFAB_ACCESS_TOKEN` 传 OAuth access token。
  这部分由于需要真实账号，只在单元测试里用模拟的 HTTP 服务验证过。
* macOS 自带的 python.org 安装版可能缺少根证书，出现 `CERTIFICATE_VERIFY_FAILED` 时运行
  `/Applications/Python 3.x/Install Certificates.command`，或 `export SSL_CERT_FILE=$(python3 -m certifi)`。

---

## AI 复刻模型并打分

`reproduce` 命令完成整条链路：取一个参考模型（本地文件或 Sketchfab）→ 渲染一张图交给 AI 图生 3D 服务（或改用文字提示词）
→ 下载 AI 生成的模型 → 自动对齐朝向 → 六视图打分。

```powershell
# 用参考模型的 3/4 视角渲染图做图生 3D（Meshy）
$env:MESHY_API_KEY = "..."
python -m src.cli reproduce --reference https://sketchfab.com/3d-models/victorian-chair-6479a1900b614b59b26784c3a7922eb3

# 改用文字提示词（文生 3D）
python -m src.cli reproduce --reference models/chair.glb --prompt "a victorian wooden dining chair with carved back"

# 用 Tripo
python -m src.cli reproduce --reference models/chair.glb --provider tripo --api-key ...

# 已经用别的工具（网页版 Meshy / Tripo / Hunyuan3D / TRELLIS ...）生成好了模型：跳过生成，只做对齐 + 打分
python -m src.cli reproduce --reference models/chair.glb --candidate downloads/ai_chair.glb

# 只调用生成，不打分
python -m src.cli generate-model --provider meshy --image renders/chair/iso.png
python -m src.cli generate-model --provider tripo --prompt "a coffee mug"
```

输出目录 `outputs/run_*/` 除了常规文件还有：

* `generation/hero_iso.png`：发给 AI 的那张图（默认 `iso` 三/四视角、1024 px、白底；`--hero-view` / `--hero-size` 可改）；
* `models.json` 里的 `generation`（服务、任务 id、耗时、生成模型路径）和 `candidate.auto_orient`（自动选出的朝向）。

生成的模型保存在 `models/generated/<provider>_<task_id>.glb`，旁边的同名 json 记录任务信息。

### 支持的服务

| `--provider` | 环境变量 | 图生 3D | 文生 3D | 接口 |
| --- | --- | --- | --- | --- |
| `meshy`（默认） | `MESHY_API_KEY` | `POST /openapi/v1/image-to-3d`（图片以 base64 data URI 发送） | `POST /openapi/v2/text-to-3d`（`preview` 阶段） | 轮询到 `SUCCEEDED` 后下载 `model_urls.glb` |
| `tripo` | `TRIPO_API_KEY` | `POST /v2/openapi/upload` + `POST /task`（`image_to_model`） | `POST /task`（`text_to_model`） | 轮询到 `success` 后下载 `output.pbr_model` |

两者都是付费 / 按额度计费的服务，只有显式执行 `generate-model` 或 `reproduce` 时才会调用。默认不生成贴图（`--texture` 开启），
因为打分只看几何。`--model-version` 选服务的模型版本，`--param key=value` 透传任意任务参数（可重复），
`--poll-interval`（默认 10 s）和 `--timeout`（默认 30 min）控制等待。

Tripo 的实测计费（`models.json` 的 `generation.meta.consumed_credit` 会记录每次实际扣除的点数，日志也会打印剩余额度）：

| 方式 | 点数 |
| --- | --- |
| 图生 3D，`--model-version v3.1-20260211`，不带贴图（推荐） | 20 |
| 图生 3D，v2.5（不指定版本时的默认）或 v1.4，即使关闭贴图 | 30 |
| 文生 3D，不带贴图（官方价目表） | 10 |

注意：Tripo v3.x 会保留输入图片的视角，用 3/4 视角图生成的模型会整体旋转约 45°；`reproduce` 默认开启的自动对齐会把它转回来。

这两个客户端按官方文档 / 官方 SDK 的接口实现，并用模拟的 HTTP 服务做了单元测试；没有用真实账号跑过，
第一次使用时如果接口有变动请把报错贴出来。

### 只看形状：`configs/shape.yaml`（`compare-models` / `reproduce` 的默认配置）

`configs/default.yaml` 下 SSIM 和 LPIPS 对整张图计算，而画面大部分是共同的白色背景，所以毫不相关的两个物体也有 60 分左右，
AI 复刻和「换了个物体」之间拉不开。`configs/shape.yaml` 做两件事：每个视图各自按前景包围盒裁剪并铺满画布
（画面里物体的大小、各视图之间的比例不再计分，只剩每个视图的形状），把权重压到轮廓和边缘上，并用 `score_floors` 扣掉不相关物体也能拿到的底分（见下）。
`compare-models` 和 `reproduce` 不传 `--config` 时就用它；图片命令 `compare` / `compare-pair` 仍用 `default.yaml`。

```powershell
python -m src.cli compare-models --reference a.glb --candidate b.glb --auto-orient            # 默认已是 shape.yaml
python -m src.cli compare-models --reference a.glb --candidate b.glb --config configs/default.yaml
```

**为什么要有 `score_floors`。** 即使只看形状，两个毫不相关的物体（椅子 vs 龙）原先也有 40 多分，因为每个指标都有一个「白送」的底分：

| 指标 | 不相关物体也能拿到 | 原因 |
| --- | --- | --- |
| SSIM | 0.80–0.87 | 白背景 + 同样的灰色着色；椅子 vs 龙 0.86，椅子 vs 自己的 AI 复刻 0.85，没有区分度，所以权重设为 0 |
| LPIPS | 距离 0.30–0.55（= 58–74 分） | `100·exp(-d)` 的映射很宽松 |
| 轮廓 IoU | 0.15–0.45 | 两个都居中、都铺满画布的图形必然重叠 |
| Edge | 8–35 分 | 轮廓线总有一部分离得不远 |

`score_floors` 把每个指标的底分减掉再拉伸回 0–100，`score_gamma` 把中段抬高一些，`view_power` 决定六个视图怎么合成一个分：

```text
x        = clip((原始分 − floor) / (100 − floor), 0, 1)
校准分   = 100 × x ^ score_gamma                      # 每个指标
视图分   = 加权平均(校准分)
综合分   = ( mean(视图分 ^ view_power) ) ^ (1 / view_power)   # 幂平均
```

* 低于 floor 记 0，100 仍是 100（同一个模型仍然满分）。
* `score_gamma < 1` 时曲线上凸，「认得出是同一个东西、但比例姿态有出入」的复刻不会被线性刻度压得太低。
* `view_power < 1` 的幂平均偏向**最差的那个视图**：真正的复刻每个视图都对得上，而不相关的物体经常只在某一个视图上碰巧像
  （椅子和茶壶从正上方看都是一个圆饼，IoU 0.8）。普通平均下这一对有 24–29 分，幂平均后是 0.2 分。
* `metrics.json` 同时保留原始分（`*_score`）和校准后的分（`calibrated_scores`），report.png 里多一行 `after floors`。
  `configs/default.yaml` 三个参数都是中性值（无 floors、gamma 1、power 1），行为不变。

#### 参数是怎么定的：`scripts/calibrate.py`

```powershell
python scripts/calibrate.py measure --workers 5    # 测 85 组样本对的原始指标，缓存到 outputs/calibration/raw.json（约 30 分钟）
python scripts/calibrate.py fit                    # 网格搜索 floors / gamma / 权重 / view_power，打印各类样本的分数分布
```

样本：Sketchfab 上 11 个类别的 27 个单体模型（椅子、角色、龙、杯子、剑、汽车、吉他、桌子、茶壶、运动鞋、战斗机），组成 85 对，不花任何生成积分：

| 类别 | 对数 | 说明 | shape.yaml 下的分数（最小 / 中位 / 最大） |
| --- | --- | --- | --- |
| identical | 4 | 模型和自己比 | 100 / 100 / 100 |
| mild | 9 | 轻微变形的副本（各轴缩放 ±4 %） | 87.9 / 93.2 / 96.2 |
| medium | 9 | 中等变形（缩放 ±12 %、弯曲、扭转 8°） | 68.3 / 76.4 / 84.6 |
| strong | 9 | 强变形（缩放 ±25 %、明显弯曲、扭转 25°） | 35.1 / 49.1 / 67.5 |
| ai_image | 2 | Tripo v3.1 图生复刻（椅子、Viking） | 86.6 / 87.5 / 88.4 |
| ai_text | 3 | Tripo v3.1 文生复刻（椅子、Viking、龙） | 29.9 / 52.3 / 54.1 |
| same | 19 | 同类别的另一个模型 | 1.1 / 45.5 / 79.4 |
| unrelated | 30 | 不同类别 | 0.1 / 3.9 / 49.1（90 % 的样本低于 36） |

选定的参数：floors silhouette 45 / edge 30 / lpips 70，`score_gamma` 0.4，`view_power` 0.25，权重 silhouette 0.45 / edge 0.25 / lpips 0.30 / ssim 0。

大致读法：85 以上是高质量复刻，50 左右是认得出的粗略复刻，10 以下是不同物体。需要知道的几点：

* 「同类别的另一个模型」分布很宽（0–79）：两把电吉他、两辆轿车本来就很像，能到 70 多分；两张造型不同的桌子只有十几分。
  文生复刻（约 52）落在这个区间里，说明它还原的是类别和大致造型，而不是这个具体模型。
* 不相关物体里仍有少数高分：剑 vs 电吉他 40 多分，因为两者在三个视图里都是一根细长条，从轮廓上确实像。
  自动对齐会给每一对都找"最像"的摆法，所以不相关物体的高分尾巴（8/30 高于 20）也是对齐之后的真实相似度；
  上表是改进自动对齐（见[渲染方式](#渲染方式)）之后重测的，旧版对齐常把不相关物体摆歪，那时只有 4/30 高于 20。
* 参数是对这 85 对拟合的，真实 AI 复刻只有 5 个；换一批很不一样的物体（比如全是细长件）时建议重跑 `calibrate.py`。
* 模型本身的尺寸在所有配置下都不影响分数（渲染前统一归一化）。

### 汇总所有运行：`scripts/summarize_runs.py`

```powershell
python scripts/summarize_runs.py                 # 扫描 outputs/run_*，写出 outputs/results.md 和 results.csv
python scripts/summarize_runs.py --sort score
```

每行一次运行：比较的两个模型、生成方式（服务 / 图生或文生 / 版本 / 消耗点数）、配置、自动对齐结果、六个视图的分数和综合分。

### 怎么解读分数

* AI 生成的模型通常比例、细节和原模型都有差异；同一参考模型下不同服务、不同版本、不同提示词之间的**相对**分数更有意义。
  实测同一把维多利亚椅（Sketchfab `6479a190…`）：Tripo v2.5 图生 3D 91.8 分、v3.1 不带贴图 91.3 分、v1.4 87.9 分。
* 自动对齐解决轴对齐和绕向上轴的任意旋转。如果生成的模型绕其他轴倾斜，或左右镜像了，分数会偏低，需要自己在建模软件里转正后用 `--candidate` 传入。
* 六个正交视图看不到内部结构和被遮挡的凹陷，贴图和颜色也不参与打分。

---

## Agent 测评（配置文件驱动，无头运行）

`python -m src.agentbench` 按一个 JSON（或 YAML）配置文件跑完整个测评：让大模型 agent 在指定的运行环境里
（Blender MCP、本项目自带的工具，或者任意别的 agent 命令行）根据提示词和参考图片建模，把它每一步的思考过程、
工具调用和结果、看到的截图都记下来，最后用上面的 `compare-models` 流程给它导出的模型打分，还可以把分数反馈给它再改一轮。
命令本身不带任何题目相关的参数：题目、模型 API、工作区全部写在配置文件里，同学们改好配置文件直接启动即可。

```bash
python scripts/make_agent_example.py                                    # （可选）重新生成示例题目：马克杯
python -m src.agentbench check benchmarks/example/mcp_blender.json      # 检查配置、参考图片、MCP 工具列表，不调用模型
python -m src.agentbench run   benchmarks/example/mcp_blender.json      # 正式运行
python -m src.agentbench run   benchmarks/example/mcp_blender.json --task mug --set llm.model=qwen3-vl-max   # 只跑一题 / 临时覆盖配置
```

Docker 里运行（Blender MCP 和 benchmark 两个容器，`workspaces/` 在两边都挂载在 `/app/workspaces`）：

```bash
docker compose up -d blender            # 已经在跑的话也要重新执行一次，让新的 workspaces 挂载生效
docker compose run --rm benchmark python -m src.agentbench run benchmarks/example/mcp_blender.json
```

### 配置文件格式

示例见 [benchmarks/example/](benchmarks/example/)：`mcp_blender.json`（Blender MCP + OpenAI 兼容接口）、
`builtin_python.json`（本项目工具 + Claude）、`command_claude_code.json`（调用外部 agent 命令行）。

```jsonc
{
  "name": "mug-blender-mcp",                 // 运行名，输出目录叫 <name>_<时间>
  "description": "",

  // 1. 运行环境类型
  "runtime": {
    "type": "mcp",                           // mcp | builtin | command
    "mcp": {
      "transport": "streamable_http",        // streamable_http | sse | stdio
      "url": "${MCP_URL:-http://localhost:8000/mcp}",
      "workspace_path": "/app/workspaces/example",   // MCP 服务器看到的工作区路径（Docker 挂载点）；本机服务器可省略
      "exclude_tools": ["*sketchfab*", "*tripo*"]    // 屏蔽的 MCP 工具（通配符），也有 include_tools
    },
    "tools": ["list_files", "read_image", "render_views"]   // 额外提供的本项目工具，见下表
  },

  // 2. 大模型 API：格式 + URL + key + 模型名
  "llm": {
    "api_format": "openai",                  // openai（/chat/completions 兼容接口）| openai_responses（OpenAI /responses）| anthropic（Claude）
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "${LLM_API_KEY}",             // 从环境变量或 .env 读，不要把 key 写进文件
    "model": "qwen3-vl-plus",
    "max_tokens": 16000
  },

  // 3. 工作区：agent 在哪个文件夹下工作，输出放在哪里
  "workspace": {
    "root": "../../workspaces/example",      // 相对配置文件所在目录
    "output_dir": "runs"                     // 相对 root；每次运行建 runs/<name>_<时间>/<题目 id>/
  },

  "limits": {"max_turns": 40, "timeout_seconds": 1800, "tool_timeout_seconds": 300},
  "evaluation": {"enabled": true, "auto_orient": true, "feedback_rounds": 1},

  // 4. 题目列表：每题一个提示词 + 参考图片位置（相对工作区，可用通配符），模型自己用 read_image 去读
  "tasks": [
    {
      "id": "mug",
      "prompt": "做一个马克杯的三维模型……",
      "reference_images": ["refs/mug/*.png"],
      "reference_model": "answers/mug.glb"   // 打分用的参考模型，相对配置文件；不给 agent 看
    }
  ]
}
```

系统提示词是内置的（[src/agentbench/runner.py](src/agentbench/runner.py) 的 `SYSTEM_PROMPT`，会根据运行环境自动加上
工作区、输出路径、Blender 用法等说明；MCP 服务器自带的 instructions 也会附上），配置里只写每道题的提示词。

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `runtime.type` | `mcp` | `mcp`：工具来自 MCP 服务器；`builtin`：只用本项目工具；`command`：运行别的 agent 命令行 |
| `runtime.tools` | 按类型 | 本项目工具：`list_files` `read_file` `write_file` `read_image` `run_python` `render_views` `generate_3d` |
| `runtime.mcp.transport` / `url` / `command` / `args` / `env` / `headers` | `streamable_http` | stdio 方式用 `command` + `args` 由 runner 启动服务器 |
| `runtime.mcp.workspace_path` | 无 | 工作区在 MCP 服务器那边的路径；设置后 `output_dir` 必须在 `root` 里面 |
| `runtime.mcp.include_tools` / `exclude_tools` | 全部 | 按名字通配符筛选 MCP 工具，例如屏蔽资产库下载，逼 agent 自己建模 |
| `runtime.command` | 无 | 字符串（走 shell）或参数列表；占位符 `{prompt}` `{prompt_file}` `{workspace}` `{output_dir}` `{output_model}` `{images}` `{task_id}` `{model}` `{base_url}` |
| `runtime.env` | `{}` | 给 `run_python` / `command` 子进程加的环境变量 |
| `runtime.generate` | Tripo v3.1，无贴图，最多 1 次 | `generate_3d` 工具的设置（`provider` `model_version` `max_calls` `allow_image` …），会消耗点数 |
| `llm.api_format` | `openai` | `openai`（`/chat/completions`，各家兼容接口）/ `openai_responses`（OpenAI 的 `/responses`；gpt-6 这类推理模型在 chat 接口上不能同时思考和调用工具，要用它）/ `anthropic` |
| `llm.base_url` / `api_key` / `model` | — | `anthropic` 格式可以不写 `base_url` |
| `llm.thinking` | Claude：`{"type":"adaptive","display":"summarized"}` | 原样传给 Claude 的 `thinking` 参数，`null` 表示不传 |
| `llm.supports_images` | `true` | 纯文本模型设为 `false`，图片就不发给它 |
| `llm.echo_reasoning` | `false` | OpenAI 兼容接口：把 `reasoning_content` 回传（DeepSeek / Kimi 的思考模式调用工具时需要） |
| `llm.extra_body` | `{}` | 原样合并进请求体，例如 `{"output_config": {"effort": "high"}}`、`{"enable_thinking": true}` |
| `limits.max_turns` | 40 | 每题最多调用模型的次数（含反馈轮） |
| `limits.timeout_seconds` | 1800 | 每题总时长 |
| `evaluation.feedback_rounds` | 0 | agent 说做完之后，把分数、各视图分数和 report.png 发回给它再改几轮 |
| `evaluation.*` | shape.yaml、auto_orient、六视图 | `benchmark_config` `views` `up` `size` `mesh_weight` `rig_weight` `feedback_images` |
| `tasks[].reference_images` | `[]` | 相对工作区的路径或通配符 |
| `tasks[].reference_model` / `reference_views` | 无 | 打分依据：参考模型，或者一个装着 `front.png` `side.png` … 的文件夹；都不给则只记录不打分 |
| `tasks[].attach_images` | `false` | 把参考图片直接放进第一条消息（没有 `read_image` 工具时自动这样做） |
| `tasks[].output_model` | `model.glb` | agent 必须把模型存到 `<题目输出目录>/<这个文件名>` |

字符串里的 `${变量}` / `${变量:-默认值}` 用环境变量替换，配置文件旁边或项目根目录的 `.env` 会先被读入；未知的键直接报错。
相对路径以配置文件所在目录为准（`output_dir` 和 `reference_images` 以工作区为准）。

### 三种运行环境

* **`mcp`**：连接 MCP 服务器（默认是 `docker compose up -d blender` 起的无头 Blender，`http://localhost:8000/mcp`），
  服务器的工具和 `runtime.tools` 里的本项目工具一起交给模型。模型用 `execute_blender_code` 建模、用 `get_viewport_screenshot`
  看效果、按系统提示词里给的**容器内路径**导出 glb；runner 在宿主机上读到同一个文件并打分。
* **`builtin`**：不需要 Blender。模型用 `run_python` 写 trimesh 脚本生成网格（`os.environ["OUTPUT_MODEL"]` 就是输出路径），
  用 `render_views` 按打分器同样的方式渲染自己的模型检查。可选 `generate_3d`（Tripo / Meshy 文生 3D，付费，受 `max_calls` 限制）。
* **`command`**：运行任意外部 agent（Claude Code、Codex CLI、同学自己写的脚本……）。runner 负责拼好提示词（含参考图片路径和输出路径）、
  通过占位符和环境变量（`OUTPUT_MODEL` `WORKSPACE` `LLM_BASE_URL` `LLM_API_KEY` `LLM_MODEL`，以及 `OPENAI_*` / `ANTHROPIC_*`）传进去，
  逐行记录 stdout / stderr（JSON 行会解析成结构化事件，例如 `--output-format stream-json`），结束后给它留下的模型打分。

### 记录下来的东西

```text
workspaces/example/runs/mug-blender-mcp_20260928_212902/
├── config.json          实际生效的配置（API key 已打码）
├── summary.json / .md   每题一行：状态、分数、轮数、工具调用次数、token、耗时
└── mug/
    ├── model.glb        agent 的最终模型
    ├── prompt.md        发给模型的系统提示词和第一条消息
    ├── transcript.jsonl 全部事件：每轮模型输出（正文、思考过程 reasoning、工具调用、token、耗时）、工具结果、评测、反馈、错误
    ├── conversation.md  同样的内容，可直接阅读：思考过程、代码、工具结果、截图
    ├── images/          模型看到的每一张图（参考图、Blender 截图、渲染图）
    ├── scripts/         run_python 执行过的脚本
    ├── eval/ eval_2/ …  每次评测的 metrics.json / report.png / renders
    └── task.json
```

思考过程：OpenAI 兼容接口记录返回的 `reasoning_content` / `reasoning`（DeepSeek、Qwen、Kimi、OpenRouter 等），
`openai_responses` 记录推理摘要（reasoning summary，模型觉得不需要思考时可能为空），
Claude 记录 thinking 块（默认 `display: "summarized"`，即模型思考的摘要）；不返回思考内容的模型只有正文。
任务状态：`completed`、`no_model`（结束了但没存模型）、`max_turns`、`timeout`、`llm_error`、`refused`、`command_failed`、`error`。
中途 Ctrl-C 也会保留已经记录的内容。全部题目 `completed` 时退出码为 0，否则为 1，配置错误为 2。

### 注意

* 参考模型（答案）放在工作区外面（示例里是 `benchmarks/example/answers/`，只挂载进 benchmark 容器，不挂进 Blender 容器）。
  本项目的文件工具会拒绝读工作区以外和答案所在的路径，但 `run_python` 和 `command` 是在本机直接执行代码，**不是沙箱**：
  正式测评请在 Docker 里跑，或者至少不要把答案放在 agent 能访问的位置。
* 示例的 `mcp_blender.json` 屏蔽了 Blender MCP 的资产库和 AI 生成工具（Sketchfab、Poly Haven、Tripo、Hyper3D……），
  否则 agent 可以直接下载一个现成的杯子。要测"会不会用资产库"时把 `exclude_tools` 删掉即可。
* 一道题里模型保存了新的 `model.glb` 才会重新打分；`feedback_rounds` 会把参考模型的渲染图（report.png）发给 agent，
  这等于给了它更多参考信息，比较不同模型时要保持这一项一致。

---

## 输入要求

* 支持格式：PNG、JPG/JPEG（可在配置中扩展 `input.extensions`）。
* 支持模式：RGB、RGBA、灰度（L）、带透明度的调色板图（P）、16 位灰度。
* 配对规则：**文件名相同（不区分大小写）**。`Front.PNG` 与 `front.png` 视为一对。
* 任意数量的图片；文件名不限于 front / side / top。
* 某张图片没有对应图片时，程序打印清晰的列表并**默认终止**；设置 `input.skip_unmatched: true` 或传 `--skip-unmatched` 可跳过。
* 每张图片在读取时都会做完整性校验；损坏或无法解码的文件会被记录为该对的 `error`，该对不计入综合分数，其余配对正常处理。

---

## 预处理流程

reference 与 candidate 使用**完全相同**的规则：

1. 修正 EXIF orientation。
2. 分离 alpha 通道，颜色统一转换为 RGB。
3. 生成前景 mask（见 `mask_mode`）。
4. 透明区域合成到纯白背景（`background_color` 可改）。
5. 裁剪（`crop_mode`）：
   * `none`：不裁剪；
   * `center_crop`：中心正方形裁剪；
   * `foreground_bbox`：按前景 mask 的包围盒裁剪，四周保留 `foreground_padding` 比例的 padding，并把物体居中。没有可靠 mask 时回退为不裁剪并给出警告。
6. **保持长宽比**缩放到 `canvas_size × canvas_size`（默认 512）的画布中并居中，空白处用背景色填充；**绝不拉伸**。
7. **对齐**（`alignment`，默认 `phase_correlation`）：把 candidate 平移到和 reference 重叠最好的位置，只平移、不缩放、不旋转，
   这样物体在画面里的位置差异不影响分数。估计出的偏移量写进 `metrics.json` 的 `alignment` 字段。
8. 预处理结果（对齐后的 candidate）保存到 `outputs/run_*/preprocessed/`，方便人工检查（包含 `*_mask.png`）。

对齐方式（`alignment`）：

| 值 | 行为 |
| --- | --- |
| `phase_correlation`（默认） | OpenCV 相位相关，按**内容**对齐：candidate 少个零件或多个零件时，主体依然对得准 |
| `centroid` | 对齐前景质心；简单，但局部形状变化会把质心拉偏 |
| `none` | 不对齐，位置偏差会被当成误差扣分 |

对齐用的前景信号：两张图都有可靠 mask 时用 mask，否则用「与背景色的距离」。两种情况下估计的偏移会被拒绝、图片保持不动：偏移超过 `alignment_max_shift`（默认画布的 50 %），或者移动后前景重叠反而变小（两个不相干的物体常常如此）。拒绝的偏移仍会记录在 `alignment` 字段里（`applied: false`）。
对齐只解决二维平移；相机角度不同造成的透视差异，二维对齐无法弥补。

前景 mask 来源（`mask_mode`）：

| 值 | 行为 |
| --- | --- |
| `alpha` | 只用 alpha 通道；没有 alpha 就没有 mask |
| `auto`（默认） | 有 alpha 用 alpha；否则用背景色阈值分割 |
| `background` | 总是用背景色阈值分割 |

阈值分割得到的 mask 只有在前景比例处于 `[mask_min_foreground_fraction, mask_max_foreground_fraction]` 之间时才视为可靠；例如一张没有纯色背景的照片会被判为**不可靠**，此时 Silhouette 指标标记为不可用，而不是给出虚假的分数。

---

## 指标说明

所有指标都在预处理后的 512×512 RGB 图上计算。

### 1. SSIM（`scikit-image`）

* `data_range=255`，`channel_axis=-1`（RGB 三通道平均），默认高斯窗（σ = 1.5）。
* 保存原始 `ssim`（理论范围 [-1, 1]）。
* `ssim_score = 100 × clip(ssim, 0, 1)`。

### 2. LPIPS（`lpips` 包）

* 默认 AlexNet backbone（可选 `vgg` / `squeeze`）。
* CUDA 可用时自动用 GPU，否则用 CPU。
* 图片转换为 `(1, 3, H, W)` float tensor 并归一化到 `[-1, 1]`。
* 保存原始 `lpips_distance`（越低越相似）。
* `lpips_score = 100 × exp(-lpips_distance)`（**不是** `1 - LPIPS`）。

### 3. Silhouette IoU

* 两张图都有可靠前景 mask 时：`silhouette_score = 100 × (交集 / 并集)`。
* 任一张没有可靠 mask，或两张 mask 都为空：`silhouette_iou` 与 `silhouette_score` 为 `null`，综合分数只用其余指标并重新归一化权重。

### 4. Edge similarity（OpenCV Canny + Chamfer 距离）

* 高斯模糊后用 Canny 提取边缘。
* 对两张边缘图分别做 distance transform，计算**对称截断 Chamfer 距离**：
  reference 每个边缘像素到 candidate 最近边缘的距离，反之亦然；距离在 `max_distance`（默认 20 px，按画布尺寸等比缩放）处截断并归一化到 [0, 1]。
* `edge_score = 100 × (1 − 0.5 × (mean_ref→cand + mean_cand→ref))`。
* 因此 1–2 像素的位置误差只会轻微扣分，而不是像直接求交集那样几乎全部失配。
* 两张图都几乎没有边缘时标记为 `null`；只有一张有边缘时为 0。

---

## 分数计算

默认权重（`configs/default.yaml`）：

```yaml
weights:
  lpips: 0.40
  ssim: 0.30
  silhouette: 0.20
  edge: 0.10
```

```text
pair_score = 0.40 × LPIPS_score + 0.30 × SSIM_score + 0.20 × Silhouette_score + 0.10 × Edge_score
```

某项指标不可用时，按剩余可用指标的权重重新归一化。例如 silhouette 不可用：

```text
available_weight = 0.40 + 0.30 + 0.10 = 0.80
pair_score = (0.40 × LPIPS_score + 0.30 × SSIM_score + 0.10 × Edge_score) / 0.80
```

实际使用的权重写在每一对的 `effective_weights` 里，不可用的指标列在 `unavailable_metrics`。

```text
overall_score = 所有有效 pair_score 的算术平均值
```

权重必须全部 ≥ 0、包含全部四个指标且总和为 1（容差 1e-6），否则程序报配置错误并退出。

所有分数为 0–100；终端和报告图片显示两位小数，`metrics.json` / `metrics.csv` 保留未四舍五入的原始浮点数。

---

## 输出文件

每次运行创建独立目录：

```text
outputs/run_YYYYMMDD_HHMMSS/
├── preprocessed/
│   ├── reference/      预处理后的 reference 图片（+ *_mask.png）
│   └── candidate/      预处理后的 candidate 图片（+ *_mask.png）
├── comparisons/        每对图片一张：reference | candidate | 差异图 | 指标
├── metrics.json        全部原始数值 + 分组分数 + 配置
├── metrics.csv         每行一对图片，然后是 __group__:<物体> 行，最后一行 __overall__
├── report_<物体>.png   每个物体一张：各视图逐行对比（多物体命名时）
└── report.png          多物体：一页汇总表；单物体：该物体的逐视图报告
                        （一张报告的行数超过 report_pairs_per_page 时才会分页为 *_page02.png）
```

`metrics.json` 结构：

```json
{
  "pairs": {
    "front.png": {
      "reference_path": "data/reference/front.png",
      "candidate_path": "data/candidate/front.png",
      "ssim": 0.91,
      "ssim_score": 91.0,
      "lpips_distance": 0.13,
      "lpips_score": 87.81,
      "silhouette_iou": 0.88,
      "silhouette_score": 88.0,
      "edge_score": 84.2,
      "edge_chamfer_ref_to_cand": 0.11,
      "edge_chamfer_cand_to_ref": 0.21,
      "pair_score": 88.17,
      "effective_weights": {"lpips": 0.4, "ssim": 0.3, "silhouette": 0.2, "edge": 0.1},
      "unavailable_metrics": [],
      "mask_source_reference": "alpha",
      "mask_source_candidate": "alpha",
      "error": null,
      "preprocessed_reference": "preprocessed/reference/front.png",
      "preprocessed_candidate": "preprocessed/candidate/front.png",
      "comparison_image": "comparisons/front.png"
    }
  },
  "overall_score": 88.17,
  "shape_score": null,        // compare-models / reproduce：图片分；mesh 为两边面数、face_ratio、score、weight
  "mesh": null,               // 纯图片的 compare 运行里两项都是 null
  "group_scores": { "mug": {"score": 96.16, "num_pairs": 3, "pairs": ["mug_front.png", "mug_side.png", "mug_top.png"]} },
  "num_pairs": 1,
  "num_valid_pairs": 1,
  "skipped_unmatched": {"reference_without_candidate": [], "candidate_without_reference": []},
  "run_dir": "outputs/run_20260912_132708",
  "configuration": { "...": "完整配置" }
}
```

`report.png` 每一行展示：reference、candidate、差异图（每像素绝对差的均值，magma 色图）、文件名、SSIM、LPIPS distance、LPIPS score、Silhouette IoU、Edge score、Pair score；标题展示 Overall score。

---

## 配置文件

完整默认配置见 [configs/default.yaml](configs/default.yaml)，每个键都有注释。常用项：

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `input.skip_unmatched` | `false` | 缺少配对时是否跳过（否则终止） |
| `preprocessing.canvas_size` | `512` | 画布尺寸 |
| `preprocessing.background_color` | `[255,255,255]` | 透明合成与填充颜色 |
| `preprocessing.crop_mode` | `none` | `none` / `center_crop` / `foreground_bbox` |
| `preprocessing.foreground_padding` | `0.10` | `foreground_bbox` 四周 padding 比例 |
| `preprocessing.mask_mode` | `auto` | 前景 mask 来源 |
| `preprocessing.alignment` | `phase_correlation` | 打分前把 candidate 平移对齐到 reference；`none` 关闭 |
| `preprocessing.alignment_max_shift` | `0.5` | 估计偏移超过画布这个比例时视为不可靠，不对齐 |
| `metrics.lpips.net` | `alex` | LPIPS backbone |
| `metrics.lpips.device` | `auto` | `auto` / `cuda` / `cpu` |
| `metrics.edge.max_distance` | `20.0` | Chamfer 截断距离（像素，按 512 画布缩放） |
| `weights.*` | 0.4/0.3/0.2/0.1 | 指标权重，总和必须为 1 |
| `mesh_complexity.weight` | 0（shape.yaml 0.15） | 面数多于参考时最多扣掉 shape 分数的比例；`free_log2` / `zero_log2` 定义免罚区间和归零点；`mode: fewer_is_better`（shape.yaml）下面数少于参考有最多 `bonus_weight`（0.05）的奖励（见[面数](#面数mesh_complexity)） |
| `rig.weight` | 0（shape.yaml 0.20） | 参考模型有骨骼时，骨骼 / 动画差异最多扣掉的比例；其余键见 [configs/shape.yaml](configs/shape.yaml) 注释（见[骨骼与动画](#骨骼与动画rig)） |
| `output.report_pairs_per_page` | `6` | 报告每页配对数 |
| `output.group_separator` | `"_"` | 按文件名前缀分组（多物体），每个物体一张报告；空字符串关闭 |

复制 `configs/default.yaml` 改成自己的文件后用 `--config` 传入。未知键会直接报错，避免拼写错误被静默忽略。

---

## 测试

```powershell
python -m pytest tests -q
```

测试全部使用程序自动生成的合成图片（`src/synthetic.py`），无需手动准备文件。覆盖内容：

* 同一张图与自身比较：SSIM、LPIPS、Edge、IoU 均接近 100。
* 改变亮度、改变颜色、平移、改变形状后分数下降；形状改变时轮廓与边缘分数明显下降。
* RGB、RGBA、灰度、JPEG、16 位、带透明的调色板图。
* EXIF orientation、透明合成、三种裁剪模式、长宽比保持。
* 缺少配对时的明确错误信息与 `skip_unmatched`。
* 损坏 / 截断文件的错误处理。
* 没有 CUDA 时 LPIPS 自动使用 CPU（通过 monkeypatch 模拟）。
* 权重不合法（总和≠1、负数、缺项、多余项、非数字）时的配置错误。
* 指标不可用时权重的动态重新归一化。
* 完整运行的所有输出文件、JSON/CSV 内容、CLI 退出码。

---

## Docker

```bash
docker build -t imgsim .                 # 只含 benchmark（CPU 版 torch，LPIPS 权重已内置）
docker run --rm imgsim                   # 跑测试
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/outputs:/app/outputs" imgsim \
  python -m src.cli compare-models --reference models/a.glb --candidate models/b.glb --auto-orient
```

也可以用 compose：`docker compose build`，然后 `docker compose run --rm benchmark python -m src.cli ...`
（`data/`、`models/`、`outputs/` 已挂载到宿主机）。要给别的工具（Blender 插件、AI 建模 agent 等）搭出来的模型打分，
让它导出 glb 放进 `models/`，再用上面的 `compare-models` 命令和参考模型比较即可。

API token（Sketchfab 下载、`reproduce` / `generate-model` 需要）：把 `.env.example` 复制为 `.env` 填好，compose 会自动读取；
也可以在宿主机 `export` 后直接透传。`.env` 已被 git 和 Docker 镜像忽略。用 `docker run` 时加 `--env-file .env`。

### 无头 Blender + BlenderMCP（给远程 agent 用）

`docker/blender/` 是第二个镜像：Debian 里的 Blender（4.3，amd64 / arm64 都有原生包）以 `blender -b` 无头方式常驻，
里面跑 [blender-mcp](https://github.com/ahujasid/blender-mcp) 的插件；同一容器里的 MCP 服务器改用 Streamable HTTP 传输，
远程 agent 直接连 `http://<主机>:8000/mcp` 就能建模，导出到 `/app/models` 的 glb 会出现在宿主机的 `models/`，再用 benchmark 打分。

```bash
docker compose build blender
docker compose up -d blender             # 端口可用 BLENDER_MCP_PORT=xxxx 改
docker compose logs -f blender           # 看到 "socket server listening" 和 "Uvicorn running" 即就绪

pip install "mcp>=1.9,<2"                   # 宿主机上装 MCP 客户端库（只为冒烟测试）
python scripts/blender_mcp_smoke.py      # 建一个杯子 -> models/mcp_smoke.glb + outputs/mcp_smoke.png
docker compose run --rm benchmark python -m src.cli compare-models \
  --reference models/ref.glb --candidate models/mcp_smoke.glb --auto-orient
```

MCP 客户端配置里只需要 URL，例如 Claude Code：`claude mcp add --transport http blender http://<主机>:8000/mcp`。

无头方式和桌面版插件的区别：

* 原插件在 `-b` 模式下拒绝启动（它靠 `bpy.app.timers` 在主线程执行命令，无窗口时 timer 不会触发）。
  `docker/blender/headless_server.py` 原样加载插件，自己在主线程循环里排空插件的命令队列，所以所有命令仍在主线程执行。
* `get_viewport_screenshot` 没有视口可截，改为从自动取景的相机渲染整个场景（Workbench，失败则退到 Cycles CPU）。
* 依赖 timer 做后续导入的集成（Hyper3D 的异步 glb 导入）不会生效；Poly Haven / Sketchfab / `execute_blender_code` 正常。
* Blender 启动时是默认场景（一个立方体、灯和相机）。让 agent 先 `bpy.ops.wm.read_factory_settings(use_empty=True)` 清空再建模。

环境变量（都在 compose 里透传，写进 `.env` 即可）：`BLENDERMCP_INTEGRATIONS`（默认 `polyhaven,sketchfab`）、
`BLENDERMCP_HYPER3D_API_KEY` / `BLENDERMCP_POLYPIZZA_API_KEY`（Sketchfab 复用 `SKETCHFAB_API_TOKEN`）、
`BLENDER_MCP_SAFE_MODE=1`（执行前校验脚本）、`MCP_ALLOWED_HOSTS`（限制 Host 头，例如 `myhost.example.com:*`）。

安全提示：MCP 端点没有鉴权，连上的人可以在 Blender 里执行任意 Python。只在内网或反向代理加了认证之后再暴露 8000 端口。

#### 在 Windows (x64) 上部署

两个镜像都是 Linux 镜像，架构跟着 build 的机器走：Mac (Apple Silicon) 上 build 出来的是 arm64，拷到 Windows 上不能用；
在 Windows 机器上重新 build 就得到 amd64 版本，Dockerfile 不用改。

1. 安装 Docker Desktop for Windows，用 WSL2 后端（Settings → General → Use the WSL 2 based engine）。
   Resources 里给 WSL 至少 4 GB 内存；两个镜像加起来约 3.5 GB 磁盘。
2. 在 PowerShell 里：

```powershell
git clone <仓库地址> Image-similarity-benchmark
cd Image-similarity-benchmark
Copy-Item .env.example .env         # 按需填 key
docker compose build                 # 同时构建 imgsim 和 imgsim-blender（第一次 10 到 20 分钟）
docker compose up -d blender         # MCP 端点: http://<这台机器的IP>:8000/mcp
docker compose logs -f blender
docker compose run --rm benchmark    # 跑 benchmark 测试
```

3. 远程 agent 从别的机器访问时，Windows 防火墙要放行 8000 端口（Docker Desktop 通常会自动加规则；没有的话
   `New-NetFirewallRule -DisplayName "blender-mcp" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow`）。

注意事项：

* `.gitattributes` 已强制 `docker/` 下的文件和 `*.sh` 用 LF 换行。如果是手动拷贝文件而不是 git clone，确认 `entrypoint.sh`
  不是 CRLF，否则容器会报 `no such file or directory` 启动失败。
* 镜像里跑的是 CPU 版 Blender 和 CPU 版 torch，Windows 主机的显卡不会被用到；无头建模和打分不需要显卡。
* 只想在 Windows 上跑 benchmark 而不用 Docker 的话，见上面的安装章节（Windows 可装 CUDA 版 torch 用 GPU 算 LPIPS）。

---

## 重要说明与局限性

请在解读分数时务必注意：

* **SSIM 和 LPIPS 衡量的是二维图片相似度。** 它们比较的是像素与神经网络特征层面的相似程度，**不能直接证明三维模型的几何是正确的**。
* **比较的两张图片必须具有相同或非常接近的视角。** 相机角度、焦距、距离不同，即使模型完全一致，分数也会很低。本项目不会尝试对齐视角。
* **图片中物体的大小、背景和光照都会影响分数。** 位置偏差默认会被对齐步骤消除（只平移，不缩放、不旋转）；大小不同可以用 `foreground_bbox` 裁剪；背景颜色或亮度不同会降低 SSIM 与 LPIPS，视角不同造成的透视差异无法用二维对齐弥补。请尽量在相同的渲染/拍摄设置下生成图片。
* **综合分数（pair_score / overall_score）是本项目自己定义的分数，不是行业统一标准。** 它只是四个指标的加权平均，方便横向对比同一套设置下的不同 candidate。
* **权重（0.40 / 0.30 / 0.20 / 0.10）是初始设定，需要通过人工评价数据进一步校准。** 建议收集一批人工打分的图片对，再调整权重使综合分数与人工判断相关性最高。
* 不同视角（不同文件名）的图片不会互相比较；overall_score 只是各对分数的平均，并不代表任何跨视角的一致性。
* **面数只是精细度的粗略代理。** 同一个物体可以用很少的面建得很准，也可以用很多面建得很糙；`mesh_score` 只惩罚数量级上的差异，
  而且完全不看拓扑质量。参考模型本身面数偏离常规时（例如扫描件），可以调低 `mesh_complexity.weight`。
* **骨骼分只看结构，不看动作内容和变形质量。** 关节数量和位置、权重覆盖、动画覆盖和运动幅度都对得上就是满分，
  蒙皮权重画得糟、动作难看都测不出来；骨骼命名也没有参与比较。
* Silhouette IoU 依赖可靠的前景 mask（alpha 通道或纯色背景）。没有可靠 mask 时该指标为 `null`，而不是伪造一个数值。
* **三维模型对比仍然是二维图片相似度。** 六视图能反映外形轮廓和大体结构，但看不到内部结构；
  两个模型的朝向、单位必须先统一（见[三维模型对比](#三维模型对比)），否则分数没有意义。渲染没有贴图和材质，只比较几何。

---

## 项目结构

```text
image_similarity_benchmark/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml          图片比较的默认配置（含注释）
│   └── shape.yaml            形状优先配置：compare-models / reproduce 的默认
├── benchmarks/example/       agent 测评的示例配置（mcp / builtin / command）和打分用的参考模型 answers/
├── workspaces/example/       agent 的工作区：参考图片 refs/，运行结果 runs/（git 忽略）
├── data/
│   ├── reference/            放 reference 图片
│   └── candidate/            放 candidate 图片
├── models/                   下载的三维模型缓存（git 忽略）；generated/ 放 AI 生成的模型
├── outputs/                  每次运行生成 run_时间/
├── scripts/
│   ├── make_sample_data.py   生成合成示例数据（默认为物体三视图）
│   ├── make_agent_example.py 生成 agent 测评的示例题目（马克杯参考模型 + 参考图片）
│   ├── summarize_runs.py     汇总 outputs/run_* 为 results.md / results.csv
│   └── calibrate.py          用 85 组样本对校准 shape.yaml 的 floors / gamma / 权重 / view_power
├── src/
│   ├── __init__.py
│   ├── cli.py                命令行入口（compare / compare-pair / compare-models / render-views / fetch-sketchfab / generate-model / reproduce）
│   ├── config.py             YAML 加载与严格校验
│   ├── preprocessing.py      读取、EXIF、alpha 合成、mask、裁剪、画布
│   ├── alignment.py          打分前的平移对齐（相位相关 / 质心）
│   ├── metrics.py            SSIM / LPIPS / Silhouette IoU / Edge / 权重合并
│   ├── benchmark.py          扫描、配对、运行、汇总
│   ├── reporting.py          metrics.json / metrics.csv / comparison / report.png
│   ├── synthetic.py          合成测试图片生成器（抽象形状 + 单物体三视图）
│   ├── objects.py            多物体测试用例：基本体拼装 + 正交三视图渲染
│   ├── render.py             三维网格加载、姿态归一化、numpy 正交光栅化六视图（含 iso 视角）
│   ├── orient.py             枚举 24 种朝向、按剪影 IoU 自动对齐 candidate
│   ├── generate.py           Meshy / Tripo 图生 3D、文生 3D 客户端（创建任务、轮询、下载）
│   ├── sketchfab.py          Sketchfab Data / Download API 客户端（下载 + 缓存）
│   └── agentbench/           配置文件驱动的 agent 测评：config / llm（OpenAI 兼容、Claude）/ tools（本项目工具、MCP）/ runner
└── tests/
    ├── conftest.py
    ├── test_preprocessing.py
    ├── test_metrics.py
    ├── test_benchmark.py
    ├── test_alignment.py     平移估计、对齐应用、阈值拒绝
    ├── test_objects.py       七个物体用例的逐指标预期与分组分数
    ├── test_render.py        光栅化、视图朝向、up 轴、CLI render-views / compare-models
    ├── test_orient.py        旋转后的模型能被自动对齐回来、iso 视角
    ├── test_generate.py      Meshy / Tripo 客户端（模拟 HTTP）、CLI generate-model / reproduce
    ├── test_agentbench.py    agent 测评：配置校验、两种 API 适配、builtin / stdio MCP / command 三种运行环境（脚本化模型）
    └── test_sketchfab.py     URL 解析、下载 / 缓存 / 错误处理（模拟 HTTP）
```
