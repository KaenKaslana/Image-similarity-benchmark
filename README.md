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
> 或 Sketchfab 上可下载的模型）用完全相同的正交相机渲染成三视图，再交给同一套流程打分；
> 渲染用纯 numpy 实现，不需要 Blender 或 OpenGL。见[三维模型对比](#三维模型对比)。

---

## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [CLI 用法](#cli-用法)
4. [三维模型对比](#三维模型对比)
5. [AI 复刻模型并打分](#ai-复刻模型并打分)
6. [输入要求](#输入要求)
7. [预处理流程](#预处理流程)
8. [指标说明](#指标说明)
9. [分数计算](#分数计算)
10. [输出文件](#输出文件)
11. [配置文件](#配置文件)
12. [测试](#测试)
13. [重要说明与局限性](#重要说明与局限性)
14. [项目结构](#项目结构)

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

`compare-models` 把「下载模型 → 渲染三视图 → 配对打分」串成一条命令：

```powershell
# 两个本地模型
python -m src.cli compare-models --reference models/a.glb --candidate models/b.glb

# 一个本地模型 vs 一个 Sketchfab 模型（需要 API token，见下文）
python -m src.cli compare-models --reference models/a.glb --candidate https://sketchfab.com/3d-models/coffee-mug-<uid>

# 只渲染三视图，不打分
python -m src.cli render-views --model models/a.glb --output renders/a --views front,side,top

# 只下载 Sketchfab 模型（缓存到 models/<uid>.glb）
python -m src.cli fetch-sketchfab https://sketchfab.com/3d-models/coffee-mug-<uid>
```

输出目录 `outputs/run_*/` 会多出：

* `renders/reference/` 与 `renders/candidate/`：渲染出的 `front.png` / `side.png` / `top.png`（RGBA、透明背景）及 `views.json`（渲染参数与网格统计）；
* `models.json`：两个模型的来源（本地路径或 Sketchfab 元数据：名称、作者、许可证）。

其余文件（`metrics.json`、`report.png` 等）与 `compare` 完全相同。

### 渲染方式

* 支持 trimesh 能读取的格式：glb / gltf / obj / stl / ply / off / 3mf / dae 等；**不支持 fbx**。
  场景中的多个部件会合并成一个网格（节点变换已应用）。
* 模型先按 `--up` / `--front` 旋转到标准姿态（+Y 向上、+Z 朝向正视图的观察者），
  再按包围盒居中并把**最大边长**缩放到 1。三个视图共用同一个比例，所以各视图的相对尺寸保持一致。
* 正交投影，无透视。视图遵循第三角投影法：`front` 从 +Z 看，`side` 从 +X 看（模型正面在图像左侧），
  `top` 从 +Y 看（模型正面在图像底部）。另有 `back` / `left` / `bottom`，`--views all` 渲染全部六个。
* 着色为平面 headlight：灰度 = 环境光 + 面法线与视线夹角，两个模型使用完全相同的光照。`--style silhouette` 输出纯黑剪影。
* 默认 2 倍超采样抗锯齿（`--supersample`），画布 512 px（`--size`），物体最大边占画布 85 %（`--fill`）。
* 渲染是纯 numpy 的 z-buffer 光栅化，8 万面的网格单个视图约 0.5 s；不需要显卡、OpenGL 或 Blender。

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--views` | `front,side,top` | 逗号分隔的视图名或 `all` |
| `--up` | `+y` | 模型的向上轴。glTF 规范是 +Y；Blender / 很多 STL 是 +Z |
| `--front` | 随 `--up` | 模型正面朝向的轴（+Y 向上时默认 +Z，+Z 向上时默认 -Y） |
| `--size` | 512 | 渲染分辨率 |
| `--fill` | 0.85 | 最大边占画布的比例 |
| `--style` | shaded | `shaded` 或 `silhouette` |

**朝向是最容易出错的地方**：两个模型如果「正面」定义不一致（一个 +Z 朝前、一个 -Y 朝前），即使模型一样分数也会很低。
三种处理方式：

* `--auto-orient`：枚举 candidate 的全部 24 种轴对齐朝向，渲染低分辨率剪影，选与 reference 三视图 IoU 平均值最高的那一种。
  选中的朝向和前几名的 IoU 写在 `models.json` 的 `candidate.auto_orient` 里。只处理 90° 的旋转，不处理任意角度倾斜和镜像。
  实测：把一把椅子绕 X、Z 各转 90° 后直接比较只有 63.8 分，加 `--auto-orient` 后恢复到 100 分。
* `--candidate-up` / `--candidate-front`：手动为 candidate 指定与 reference 不同的轴。
* 先用 `render-views` 分别看一眼三视图，再决定参数。

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
→ 下载 AI 生成的模型 → 自动对齐朝向 → 三视图打分。

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
因为打分只看几何。`--poll-interval`（默认 10 s）和 `--timeout`（默认 30 min）控制等待。

这两个客户端按官方文档 / 官方 SDK 的接口实现，并用模拟的 HTTP 服务做了单元测试；没有用真实账号跑过，
第一次使用时如果接口有变动请把报错贴出来。

### 怎么解读分数

* AI 生成的模型通常比例、细节和原模型都有差异，分数落在 50–80 分是正常的；同一参考模型下不同服务、不同提示词之间的**相对**分数更有意义。
* 自动对齐只解决 90° 旋转。如果生成的模型是斜着的，或左右镜像了，分数会偏低，需要自己在建模软件里转正后用 `--candidate` 传入。
* 三视图看不到内部结构，贴图和颜色也不参与打分。

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
├── report_<物体>.png   每个物体一张：三视图逐行对比（多物体命名时）
└── report.png          多物体：一页汇总表；单物体：该物体的三视图报告
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

## 重要说明与局限性

请在解读分数时务必注意：

* **SSIM 和 LPIPS 衡量的是二维图片相似度。** 它们比较的是像素与神经网络特征层面的相似程度，**不能直接证明三维模型的几何是正确的**。
* **比较的两张图片必须具有相同或非常接近的视角。** 相机角度、焦距、距离不同，即使模型完全一致，分数也会很低。本项目不会尝试对齐视角。
* **图片中物体的大小、背景和光照都会影响分数。** 位置偏差默认会被对齐步骤消除（只平移，不缩放、不旋转）；大小不同可以用 `foreground_bbox` 裁剪；背景颜色或亮度不同会降低 SSIM 与 LPIPS，视角不同造成的透视差异无法用二维对齐弥补。请尽量在相同的渲染/拍摄设置下生成图片。
* **综合分数（pair_score / overall_score）是本项目自己定义的分数，不是行业统一标准。** 它只是四个指标的加权平均，方便横向对比同一套设置下的不同 candidate。
* **权重（0.40 / 0.30 / 0.20 / 0.10）是初始设定，需要通过人工评价数据进一步校准。** 建议收集一批人工打分的图片对，再调整权重使综合分数与人工判断相关性最高。
* 不同视角（不同文件名）的图片不会互相比较；overall_score 只是各对分数的平均，并不代表任何跨视角的一致性。
* Silhouette IoU 依赖可靠的前景 mask（alpha 通道或纯色背景）。没有可靠 mask 时该指标为 `null`，而不是伪造一个数值。
* **三维模型对比仍然是二维图片相似度。** 三视图能反映外形轮廓和大体结构，但看不到被遮挡的内部结构；
  两个模型的朝向、单位必须先统一（见[三维模型对比](#三维模型对比)），否则分数没有意义。渲染没有贴图和材质，只比较几何。

---

## 项目结构

```text
image_similarity_benchmark/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml          默认配置（含注释）
├── data/
│   ├── reference/            放 reference 图片
│   └── candidate/            放 candidate 图片
├── models/                   下载的三维模型缓存（git 忽略）；generated/ 放 AI 生成的模型
├── outputs/                  每次运行生成 run_时间/
├── scripts/
│   └── make_sample_data.py   生成合成示例数据（默认为物体三视图）
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
│   ├── render.py             三维网格加载、姿态归一化、numpy 正交光栅化三视图（含 iso 视角）
│   ├── orient.py             枚举 24 种朝向、按剪影 IoU 自动对齐 candidate
│   ├── generate.py           Meshy / Tripo 图生 3D、文生 3D 客户端（创建任务、轮询、下载）
│   └── sketchfab.py          Sketchfab Data / Download API 客户端（下载 + 缓存）
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
    └── test_sketchfab.py     URL 解析、下载 / 缓存 / 错误处理（模拟 HTTP）
```
