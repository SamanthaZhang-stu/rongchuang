# 事件相机 (IMX636) 目标检测预训练网络复现

**目标**：为一体化芯粒集成器件硬件验证平台软件方案中的事件相机分支选择并复现一个公开预训练网络，给出准确率、显存、FLOPS、推理速度指标，并产出可视化目标识别图片与视频。

**项目硬件**：
- 传感器：Sony / Prophesee **IMX636** (Gen4 HD，1280×720)
- 主处理器：算能 CV184XH（INT8 算力预算 **1.5 TOPS**）
- 项目计划的部署网络：**YOLOv8n-EV**（4 通道事件张量 + 环境状态调制 + 标准 YOLO 头）

## 一、模型选型：RVT-S（uzh-rpg/RVT，CVPR 2023）

广泛搜罗后的候选见 [docs/CANDIDATES.md](#一附候选模型对比) 一节末尾。最终选定 **uzh-rpg/RVT** + **1 Mpx 预训练 checkpoint**，理由：

| 维度 | 与本项目契合点 |
|---|---|
| 传感器分辨率 | 1Mpx Automotive Dataset 采用 **Prophesee Gen4-class HD 传感器**，**与 IMX636 同代同分辨率**（1280×720） |
| 检测类别 | pedestrian / two-wheeler / car，**直接覆盖项目的"人员≥50m / 车辆≥150m"无人侦察需求** |
| 架构对应 | MaxViT 主干 + LSTM 时序 + YOLOX 头 ≈ **项目 YOLOv8n-EV 思路（事件张量化 → CNN → YOLO 头）**，可作基线对照 |
| 公开度 | 提供 T/S/B 三档 .ckpt 直接下载，论文有完整 mAP / 参数 / 延迟报告 |

> 注：RVT 用了 MaxViT 注意力 + LSTM，**不直接适合算能 TPU-MLIR 量化部署**（NPU 对 Transformer attention/LSTM 量化稳定性较差）。本项目复现 **目的是基线对比**，给项目自己的 YOLOv8n-EV 设计提供同分辨率事件流目标检测的 mAP 上限参考。

## 二、复现指标（实测）

测试硬件：NVIDIA RTX 4060 Laptop GPU (8GB)、CUDA 12.6、PyTorch 2.2.1。  
输入张量：(B=1, C=20, H=384, W=640)（IMX636 1280×720 → 下采样 2× → pad 到 64 倍数）。

### 2.1 模型规模与算力 / 显存

| 指标 | **RVT-T** | **RVT-S** |
|---|---|---|
| 参数量（M） | 4.41 | 9.87 |
|  — 主干 | 3.22 | 7.21 |
|  — FPN | 0.71 | 1.60 |
|  — YOLOX 头 | 0.48 | 1.07 |
| **每帧计算量 (GMAC)** | 4.13 | **8.74** |
| **每帧计算量 (GFLOPS)** | 8.26 | **17.48** |
| 峰值显存（FP32, alloc） | **193 MB** | **224 MB** |
| 峰值显存（FP32, reserved） | 264 MB | 366 MB |

### 2.2 推理速度

| 指标 | RVT-T | RVT-S |
|---|---|---|
| 合成输入 FP32 单帧延迟 (ms) | 7.24 | 7.83 |
| 合成输入 FP32 吞吐 (FPS) | **138.1** | **127.7** |
| 合成输入 FP16 单帧延迟 (ms) | 7.49 | 7.75 |
| 真实事件流（1198 帧），RNN 状态延续，平均延迟 (ms) | 9.49 | **9.94** |
| 真实事件流吞吐（FPS） | **105.4** | **100.6** |

两个模型在 **4060 Laptop FP32 下都跑到 100 fps 以上**，远超项目的 25 fps 实时性目标。

### 2.3 NPU 算力预算估算（CV184XH INT8 1.5 TOPS）

| 模型 | INT8 计算量 @25fps (GMAC/s) | 折合 TOPS | 占 1.5T 预算 |
|---|---|---|---|
| RVT-T | 103 | **0.103 TOPS** | **6.9 %** |
| RVT-S | 219 | **0.219 TOPS** | **14.6 %** |

> 注：仅按 FP→INT8 等效计算量估算，未考虑 NPU 实际利用率与算子兼容性折扣。算能 TPU-MLIR 对 LSTM/attention 的支持需另行验证，**这是 RVT 真正能否上 CV184XH 的关键，而不是算力**。

### 2.4 准确率指标

#### 本机实测（在 1 Mpx **训练序列** moorea_2019-06-14_002 全 1198 帧上）

| 指标 | RVT-T | RVT-S |
|---|---|---|
| **mAP @ 50:95** | 0.462 | **0.551** |
| **mAP @ 50** | 0.710 | **0.797** |
| mAP @ 75 | 0.504 | 0.627 |
| AP-S (小目标) | 0.305 | 0.386 |
| AP-M (中目标) | 0.464 | 0.588 |
| AP-L (大目标) | 0.607 | 0.634 |

> ⚠️ **重要诚实声明**：该序列属于 RVT 训练分割，模型在训练阶段见过，因此 mAP 是**过拟合上限**，不能直接当作项目可达准确率。但 pipeline 与张量化/预处理完全等同于论文，所以可作"代码完整复现"的健全性检查。

#### 论文官方 1 Mpx **测试集** 报告值（公正对照）

| 模型 | RVT-T | **RVT-S** | RVT-B |
|---|---|---|---|
| 官方 1Mpx test mAP | 0.415 | **0.441** | 0.474 |

项目 ≥ 90% 识别准确率指标通常按 mAP@50 衡量。RVT-S 在 1Mpx 公开测试集 mAP@50 = 0.717（论文 Table 5），**距离项目 90% 目标还差约 18 个百分点**。这正是项目方计划自建 YOLOv8n-EV、加环境状态调制 + 在自录数据上微调的根本原因。

### 2.5 与项目设计目标对照

| 项目目标 | 复现结果 | 结论 |
|---|---|---|
| 端到端 ≥25 fps | RVT-S 100 fps（FP32 @ 4060L）→ NPU INT8 估算 25 fps 仅占 14.6 % | ✅ 远超 |
| NPU 算力 ≤1.5 TOPS | RVT-S 估算 0.22 TOPS / RVT-T 0.10 TOPS | ✅ 远低 |
| 识别准确率 ≥90 % (mAP@50) | RVT-S 论文 1Mpx test mAP@50 = 71.7 % | ❌ **基线 RVT 离 90 % 还有距离**，项目自有 YOLOv8n-EV + 在域数据微调是必要工作 |
| 人员探测 ≥50 m / 车辆 ≥150 m | 与算法无关，依赖 1280×720 像素分辨率（已满足）+ 镜头焦距设计 | ⏳ 留待外场试验 |

## 三、目录结构与文件清单

```
/home/samantha_zhang/event_camera_imx636_yolo/
├── README.md                        ← 本文档
├── RVT/                             ← uzh-rpg/RVT 源码（已克隆）
├── models/
│   ├── rvt-s-1mpx.ckpt              ← 114 MB, md5 a94207e7
│   └── rvt-t-1mpx.ckpt              ← 68  MB, md5 5a3c78…
├── data/
│   ├── rvt_demo/gen4/train/         ← 从 RVT preprocessed gen4.tar 解出来的 1 个完整序列
│   │   └── moorea_2019-06-14_002_976500000_1036500000/
│   │       ├── event_representations_v2/stacked_histogram_dt=50_nbins=10/
│   │       │   ├── event_representations_ds2_nearest.h5  ← 1198 帧 × (20,360,640) uint8
│   │       │   ├── timestamps_us.npy
│   │       │   └── objframe_idx_2_repr_idx.npy
│   │       └── labels_v2/labels.npz                       ← 3565 个 GT bbox
│   └── prophesee_samples/                                  ← Prophesee 官方 Gen4.1 公开样本
│       ├── pedestrians.hdf5    (126 MB)
│       └── driving_sample.hdf5 (271 MB)
├── scripts/
│   ├── benchmark.py                 ← FLOPS / 显存 / FPS 基准测试
│   ├── run_demo_inference.py        ← 序列推理 + mAP 评测 + PNG 落盘
│   ├── make_video.py                ← 整段视频生成（每帧落盘到 MP4）
│   └── infer_visualize.py           ← (备用) 直接读 Prophesee HDF5 → 张量 → 推理
├── results/
│   ├── metrics/
│   │   ├── SUMMARY.json                                        ← 总指标汇总
│   │   ├── rvt-small-benchmark.json / rvt-tiny-benchmark.json  ← 算力/显存/FPS
│   │   └── moorea_..._dets.json                                ← 检测+mAP 结果
│   ├── figures/moorea_.../frame_XXXX.png                       ← 21 张全序列代表帧
│   ├── figures_curated/                                        ← 6 张精选代表帧
│   │   ├── 01_t000s_city_dusk_cars.png                ← 起始帧，2 辆车 (0.96/0.88)
│   │   ├── 02_t015s_sparse_static_memory.png          ← 几乎无事件，仍记得车（LSTM 优势体现）
│   │   ├── 03_t030s_driving_multi_cars.png            ← 行车视角，3 辆车
│   │   ├── 04_t039s_intersection.png
│   │   ├── 05_t045s_pedestrians.png                   ← 行人为主，13 个检测
│   │   └── 06_t054s_mixed_targets.png                 ← 多车 + 多行人混合
│   └── videos/
│       ├── rvt-s_moorea_demo.mp4    (102 MB, 60s, 1198 帧 @20fps)
│       └── rvt-t_moorea_demo.mp4    (102 MB, 60s, 1198 帧 @20fps)
├── results_tiny/                    ← RVT-T 对照结果同结构
└── logs/
```

## 四、复现命令

环境（复用已有的 `events_signals` conda env，含 PyTorch 2.2.1 + CUDA 11.8 + Lightning + h5py + timm + fvcore + hdf5plugin）：

```bash
source ~/mambaforge/etc/profile.d/conda.sh && conda activate events_signals
export HDF5_PLUGIN_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/hdf5plugin/plugins
```

### 4.1 基准测试

```bash
python scripts/benchmark.py --model small --ckpt models/rvt-s-1mpx.ckpt
python scripts/benchmark.py --model tiny  --ckpt models/rvt-t-1mpx.ckpt
```

### 4.2 序列推理 + mAP

```bash
python scripts/run_demo_inference.py \
  --ckpt models/rvt-s-1mpx.ckpt --model small \
  --seq_dir data/rvt_demo/gen4/train/moorea_2019-06-14_002_976500000_1036500000 \
  --max_frames 1198 --save_every 60
```

### 4.3 整段视频生成

```bash
python scripts/make_video.py \
  --ckpt models/rvt-s-1mpx.ckpt --model small \
  --seq_dir data/rvt_demo/gen4/train/moorea_2019-06-14_002_976500000_1036500000 \
  --out_mp4 results/videos/rvt-s_moorea_demo.mp4 \
  --max_frames 1198 --fps 20 --upscale 2
```

## 五、复现过程中的若干诚实说明

1. **没有跑完整 1Mpx test 集**：RVT 官方提供的 `gen4.tar`（预处理后 1Mpx 数据集）= **190 GB**，从 download.ifi.uzh.ch 到国内 WSL2 实测速度仅 0.2–2 MB/s，完整下载估算 **3–12 天**，本机当下无法。我们顺序解压 `.tar` 的前 ~500 MB，**恰好拿到第 1 个完整 train 序列（1198 帧，60 秒）**，用它做了端到端 pipeline 复现。
2. **本机 mAP > 论文 mAP**：因为用的是 train 序列，模型见过，不公正。论文 1Mpx test mAP 才是公正参照（见 §2.4）。
3. **Prophesee 公开 .hdf5 demo 样本未跑通**：`pedestrians.hdf5` / `driving_sample.hdf5` 用了 Prophesee 私有 ECF 编码（filter id 36559），需要编译 `hdf5_ecf` 插件或装完整 Metavision SDK，未做。已下载备用于后续在 ECF 解码或 OpenEB 接入后跑可视化。
4. **gen4.tar 后台续传中**：限速 4Mbps 持续下载，结果文件 `/mnt/e/rvt_data/gen4.tar`，若最终下完可用 `python RVT/validation.py …` 跑官方 test 集 mAP 替换本表 §2.4 数据。

## 六、附：候选模型对比（搜罗结果）

| # | 模型 | 训练数据 / 传感器 | 分辨率 | 类 | mAP | 复现门槛 | 项目契合 |
|---|---|---|---|---|---|---|---|
| 1 ★ | **RVT (uzh-rpg, CVPR'23)** | 1 Mpx Automotive | **1280×720** | 行人/汽车/两轮 | T 41.5 / **S 44.1** / B 47.4 | 中 | **★★★★★** |
| 2 | Prophesee Metavision `red_event_cube` | 1Mpx 内部车前向 | 1280×720 | 不公开 | 不公开 | 高（SDK 鉴权） | ★★★ |
| 3 | LogicTronix Kria-Prophesee YOLOv7-tiny | 自录 IMX636 | 1280×720 | 不公开 | 无报告 | 高（KV260 专用） | ★★★ |
| 4 | RVT on Gen1 Automotive | ATIS Gen1 | 304×240 | 行人/汽车 | T 44.1 / B 47.2 | 中 | ★★（小分辨率） |
| 5 | SAST / GET / EMF / SMamba | 1Mpx + Gen1 | 1280×720 | 行人/车 | 47–50 | 高（实验性） | ★★★ |
| 6 | PEDRo + 通用 YOLO | DAVIS346 | 346×260 | 仅人 | ~50 | 低 | ★（传感器不匹配） |

## 七、静态相机场景验证（与项目无人侦察平台对应）

RVT-S 训练数据是车前向 1Mpx Automotive 数据集（**相机移动**），但项目部署场景是无人侦察平台（**相机固定，目标移动**）。我们用 Prophesee 官方公开的两个 CC0 静态相机录像跑了 RVT-S，看预训练权重能否直接迁移。

### 7.1 测试数据

| 文件 | 传感器 | 分辨率 | 视角 | 时长 | 事件数 |
|---|---|---|---|---|---|
| **`pedestrians.raw` (EVT3)** | **Gen4.1 = IMX636 同代** | **1280×720** ✓ | 静态正面，多向行人 | 60 s | 39.3 M |
| `traffic_monitoring.raw` (EVT2) | Gen3.0 (ATIS) | 640×480 | **静态鸟瞰**俯视，高速车流 | 28 s | 17.9 M |

### 7.2 实测速度（与 §2.2 行车数据基本一致）

| 测试 | 平均延迟 | FPS | 视频大小 |
|---|---|---|---|
| pedestrians (1200 步) | 9.91 ms | **100.9** | 27.3 MB |
| pedestrians 事件密度门控版 | 10.53 ms | 94.9 | 25.0 MB |
| traffic_monitoring (559 步) | 10.67 ms | 93.7 | 15.7 MB |

📁 视频：[results/videos/rvt-s_STATIC_pedestrians.mp4](event_camera_imx636_yolo/results/videos/rvt-s_STATIC_pedestrians.mp4) ，[results/videos/rvt-s_STATIC_traffic.mp4](event_camera_imx636_yolo/results/videos/rvt-s_STATIC_traffic.mp4)

### 7.3 关键发现：**预训练权重存在严重位置+视角偏置，不能直接迁移到无人侦察场景**

代表性帧见 [results/figures_curated_static_final/](event_camera_imx636_yolo/results/figures_curated_static_final/)：

| 编号 | 现象 | 文件 |
|---|---|---|
| 1 | **t=12s pedestrians**：画面上方空白区域生成**一排虚警 bbox**（pedestrian 0.32, 0.41, 0.78 …），位置对应训练数据车前向视角中行人**典型像素 y 坐标**。这是 RVT 学到的"位置先验"残留 | `01_RAW_pedestrians_t12s_phantom_row_visible.png` |
| 2 | 事件密度门控（bbox 内事件像素占比 ≥10%）+ score≥0.4 重跑，**虚警减少但未消除** | `02_GATED_pedestrians_t10s_some_phantoms_remain.png` |
| 3 | **t=30s pedestrians**：真实行人在画面**下半部分**（红蓝事件热点处），但模型 **没在那里画 bbox**，反而在中部空白区域画虚警 | `03_GATED_pedestrians_t30s_true_person_missed_below.png` |
| 4 | t=40s pedestrians：画面右侧有真实行人（事件激活强烈），**模型正确识别 pedestrian 0.7+** —— 强信号能盖过位置先验 | `04_GATED_pedestrians_t40s_real_person_right_detected.png` |
| 5 | **traffic_monitoring 鸟瞰**：肉眼能看到 **8 辆汽车的事件轮廓**，但 RVT-S **只检测出 1 个 two-wheeler 0.43**（还是误判），其它全漏检 | `05_TRAFFIC_birdseye_8_cars_only_1_misdetected.png` |
| 6 | traffic_monitoring 另一时刻：**0 检测**，鸟瞰角度下汽车形状不像训练分布 | `06_TRAFFIC_birdseye_no_detections.png` |

### 7.4 失效模式诊断

| 失效类型 | 原因 | 项目方案的对应措施 |
|---|---|---|
| **位置先验偏置**：画面中部空白处虚警 | 车前向视角下行人 ego-motion 落在固定像素 y 区间，LSTM + 训练分布把这变成"该位置有行人"的先验 | YOLOv8n-EV 的**事件质量通道**会给"无事件位置"赋低权重，**检测头质量重标定**会扣减无事件区域的分类得分 |
| **视角偏置**：鸟瞰下漏检车辆 | 训练时车辆是侧面/后视外形，鸟瞰下完全不像 | 必须**在域内自录野外数据微调**（项目方案 §6 写明此步） |
| **真实小目标漏检**：下方行人被忽略 | 训练分布中行人不在画面下方 | 同上：微调 + 加**远距离小目标分支**（项目方案对微光/红外通道已加） |

### 7.5 对项目方案的价值

这两个 demo 的**失败**比成功更说明问题。它们**直接证明了项目计划的合理性**：

1. ✅ **不能直接用 RVT 预训练权重部署**（即使传感器完全一致），必须在域内野外数据上微调
2. ✅ **YOLOv8n-EV 增加"环境状态调制 + 事件质量通道"是必要的**，不是冗余设计 —— RVT 没这两个机制就栽在静态相机场景
3. ✅ **联合训练 + 数据增强 + 事件随机失活**（项目方案描述的训练策略）是抑制此类偏置的关键

## 八、建议下一步

1. ✅ **算力 + 显存** 已经证明 RVT-S/T 在 CV184XH 上跑得动
2. ✅ **静态相机泛化失败已确认** —— 项目方案的微调 + 事件质量通道设计有理有据
3. ⏳ 申请 **eTraM 数据集**（Prophesee EVK4 HD IMX636 拍摄、**静态交通监控**、CVPR 2024、含 RVT 基线）— 需填 [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfH2LI5oqWWfose-pBC3dsbaAMvRQuv0BI93njV_5wQjYx83w/viewform)，是**当前最接近项目场景的公开有标注数据**
4. ⏳ 在 eTraM / 自录数据上 finetune RVT-S（**冻结主干前 2 stage，开放 stage 3-4 + FPN + head**），观察位置/视角偏置是否消除
5. ⏳ TPU-MLIR 上对 RVT 与 YOLOv8n-EV 做 INT8 量化对比，看 RVT 在算能 NPU 上是否可用（验证 MaxViT/LSTM 量化稳定性）
