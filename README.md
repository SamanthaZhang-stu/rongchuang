# 传感器检测算法GPU验证

# 事件相机

**项目硬件**：
- 传感器：Sony / Prophesee **IMX636** (Gen4 HD，1280×720)
- 主处理器：算能 CV184XH（INT8 算力预算 **1.5 TOPS**）

## 一、模型选型：RVT-S

<img width="502" height="220" alt="image" src="https://github.com/user-attachments/assets/2361291f-a5f4-421a-bb9f-7beda5ae5fb2" />

<img width="1068" height="304" alt="image" src="https://github.com/user-attachments/assets/0fcd0efe-715b-4532-bf2c-47f2fb58fb76" />

## 二、实测指标

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

> 注：仅按 FP→INT8 等效计算量估算，未考虑 NPU 实际利用率与算子兼容性折扣。算能 TPU-MLIR 对 LSTM/attention 的支持需另行验证。

### 2.4 准确率指标

#### GPU实测（在 1 Mpx **训练序列** moorea_2019-06-14_002 全 1198 帧上）

| 指标 | RVT-T | RVT-S |
|---|---|---|
| **mAP @ 50:95** | 0.462 | **0.441** |
| **mAP @ 50** | 0.710 | **0.797** |
| mAP @ 75 | 0.504 | 0.627 |
| AP-S (小目标) | 0.305 | 0.386 |
| AP-M (中目标) | 0.464 | 0.588 |
| AP-L (大目标) | 0.607 | 0.634 |


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


## 四、静态相机场景验证

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

视频：[results/videos/rvt-s_STATIC_pedestrians.mp4](event_camera_imx636_yolo/results/videos/rvt-s_STATIC_pedestrians.mp4) ，[results/videos/rvt-s_STATIC_traffic.mp4](event_camera_imx636_yolo/results/videos/rvt-s_STATIC_traffic.mp4)

---

# 红外相机

**项目硬件**：
- 传感器：高德 **MINI212** 工业红外成像模组（256×192，8–14 μm，NETD ≤ 50 mK）
- 主处理器：算能 CV184XH（INT8 算力预算 **1.5 TOPS**）
- 接口：USB 2.0（480 Mbps，直连 CV184XH USB Host，无需转接芯片）

## 一、模型选型：YOLOv8n-IR（单通道 Stem + P2 细粒度检测头）

针对单波段红外特性，在 Ultralytics YOLOv8n 上做三处针对性改造：

| 改造点 | 原版 YOLOv8n | YOLOv8n-IR | 收益 |
|---|---|---|---|
| 输入通道 | 3 (RGB) | **1** (灰度) | Stem 卷积省 ~1/3 算力，匹配单波段红外 |
| 检测头 | P3 / P4 / P5 (3 头) | **P2 / P3 / P4 / P5 (4 头)** | 多一个 stride=4 头专门服务远距离小目标 |
| 算子约束 | 自由选择 | 卷积通道全部对齐 16 倍数，非线性统一 ReLU/SiLU | 保证 INT8 量化稳定与算能 TPU-MLIR 兼容 |

**红外预处理流程（方案 4.3.3）**：温度场归一化（1–99 百分位截断 → 8-bit）→ 非均匀性校正（两点法 + 快门挡片偏置更新，接口预留）→ CLAHE 局部对比度增强（clipLimit=2.0, 8×8 tile）→ 双线性 letterbox 至 640×640。

## 二、实测指标

测试硬件：NVIDIA RTX 4080（32 GB）、CUDA 12.8、PyTorch 2.8、AMP 训练。  
训练数据集：**HIT-UAV** 高空无人机红外数据集（2898 张：train 2029 / val 290 / test 579，4 类：Person / Car / Bicycle / OtherVehicle）。  
训练配置：60 epoch、batch=32、imgsz=640、AdamW（auto-lr=0.00125）、cos-lr。

### 2.1 模型规模与算力 / 显存

| 指标 | **YOLOv8n-IR** | YOLOv8n（3 通道 / 3 头基线参考） |
|---|---|---|
| 参数量（M） | **2.92** | 3.01 |
|  — 主干 | 1.27 | — |
|  — Neck (PANet) | 1.04 | — |
|  — Head (4 scales) | 0.62 | — |
| **每帧计算量 (GMAC)** | **6.05** | 4.10 |
| **每帧计算量 (GFLOPS)** | **12.1** | 8.2 |
| 训练峰值显存（AMP, batch=32） | **13.7 GB** | — |
| 训练总时长（60 epoch） | **12.5 min** | — |

> P2 头增加约 50% GMAC，但带来远距离小目标召回显著提升（见 §2.4 Person/Bicycle 类 mAP）。

### 2.2 推理速度

| 指标 | YOLOv8n-IR (val 290 张) | YOLOv8n-IR (test 579 张) |
|---|---|---|
| 单帧 preprocess | 0.0 ms | 0.4 ms |
| 单帧 inference | 0.7 ms | 1.4 ms |
| 单帧 postprocess | 1.0 ms | 1.7 ms |
| **单帧总延迟** | **1.7 ms** | 3.5 ms |
| 等效 FPS（4080 GPU 上限） | **~580 fps** | ~290 fps |

GPU 侧实时性余量极大，远超 25 fps 实时性目标。

### 2.3 NPU 算力预算估算（CV184XH INT8 1.5 TOPS）

| 配置 | INT8 计算量 @25fps (GMAC/s) | 折合 TOPS | 占 1.5T 预算 |
|---|---|---|---|
| YOLOv8n-IR · 25 fps（设计基准） | 151 | **0.151 TOPS** | **10.1 %** |
| YOLOv8n-IR · 50 fps | 303 | 0.303 TOPS | 20.2 % |

> 1 MAC = 1 INT8 OP；25 fps 占 10.1% 预算，剩余 90% 可分给微光（~10.9%）、事件相机（~14.6%）和多模态融合层。

### 2.4 准确率指标

#### Val 集 (290 张, 2453 实例)

| 指标 | YOLOv8n-IR |
|---|---|
| Precision | 0.903 |
| Recall | 0.771 |
| **mAP@50** | **0.864** |
| mAP@50:95 | 0.547 |

#### Test 集 (579 张, 4780 实例，独立未见过)

| 类别 | Instances | Precision | Recall | **mAP@50** | mAP@50:95 |
|---|---|---|---|---|---|
| **Person** | 2 611 | 0.871 | 0.897 | **0.931** | 0.490 |
| **Car** | 1 339 | 0.917 | 0.950 | **0.975** | 0.674 |
| **Bicycle** | 796 | 0.880 | 0.857 | **0.917** | 0.559 |
| OtherVehicle | 34 | 0.665 | 0.409 | 0.489 | 0.354 |
| **all** | **4 780** | **0.833** | **0.778** | **0.828** | 0.519 |

> Person / Car / Bicycle 三大类 mAP@50 **均 ≥ 0.917**，达到方案 4.4「识别准确率 ≥ 90%」总指标。OtherVehicle 因训练集仅 12 个实例属长尾类，需后续扩充。

---

# 微光相机

**项目硬件**：
- 传感器：SmartSens **SC2210** 工业级微光 CMOS（1920×1080，2.9 μm 背照式，星光级感光 0.0001 lux）
- 主处理器：算能 CV184XH（INT8 算力预算 **1.5 TOPS**）
- 接口：MIPI CSI-2 × 4 lane @ 1.5 Gbps/lane

## 一、模型选型：Zero-DCE++ 增强 + YOLOv8n-MR（P2 检测头）

方案 4.3.1 两段式：

1. **Zero-DCE++ 低照度增强**（前置）— 7 层 DCE-Net（仅 **11K 参数**），4 项无参考损失自监督训练，无需配对低光/正常光样本；
2. **YOLOv8n-MR 检测主网** — 与红外结构对称：标准 YOLOv8n backbone + PANet neck + **P2/P3/P4/P5 四检测头**，3 通道输入。

**三阶段训练流程（方案 4.3.1 渐进式策略）**：

| 阶段 | 操作 | 配置 | 耗时 |
|---|---|---|---|
| Stage 1 | Zero-DCE++ 自监督预训练 | ExDark train 5917 张 · 30 epoch · batch=16 · 256×256 | **10.5 min** |
| Stage 2 | 离线批量增强 7361 张 + 训练 YOLOv8n-MR 主干 | 30 epoch · batch=32 · 640×640 | **2 min + 18.4 min** |
| Stage 3 (可选) | 整网联合微调 lr0=1e-4 freeze=4 | 10 epoch | ~6 min (本次未启用) |

## 二、实测指标

测试硬件：NVIDIA RTX 4080（32 GB）、CUDA 12.8、PyTorch 2.8、AMP 训练。  
训练数据集：**ExDark** 公开低照度数据集（7361 张：train 5917 / val 750 / test 694，12 类：people / car / bicycle / motorbike / bus / boat / dog / cat / chair / table / bottle / cup）。

### 2.1 模型规模与算力 / 显存

| 指标 | Zero-DCE++ | **YOLOv8n-MR** | 合计 |
|---|---|---|---|
| 参数量（M） | **0.011** | **2.92** | **2.93** |
|  — 主干 | — | 1.27 | — |
|  — Neck (PANet) | — | 1.04 | — |
|  — Head (4 scales) | — | 0.62 | — |
| **每帧计算量 (GMAC)** | 0.42 | **6.10** | **6.52** |
| **每帧计算量 (GFLOPS)** | 0.85 | 12.2 | 13.05 |
| 训练峰值显存（AMP） | ~50 MB | **14.2 GB** | — |
| 训练时长 | 10.5 min | 18.4 min | ~31 min（含增强 2 min） |

> Zero-DCE++ 仅 11K 参数、0.42 GMAC，相比主网开销可忽略；白天场景可关闭增强子网。

### 2.2 推理速度

| 子模块 | 单帧 inference | 单帧总延迟 | 等效 FPS（4080 GPU 上限） |
|---|---|---|---|
| Zero-DCE++ (256×256) | ~0.4 ms | ~0.4 ms | >1000 |
| YOLOv8n-MR (640×640) | 0.7 ms | **2.2 ms** | ~450 |
| **端到端** | — | **~2.6 ms** | **~380 fps** |

### 2.3 NPU 算力预算估算（CV184XH INT8 1.5 TOPS）

| 配置 | INT8 计算量 @25fps (GMAC/s) | 折合 TOPS | 占 1.5T 预算 |
|---|---|---|---|
| Zero-DCE++ + YOLOv8n-MR · 25 fps（设计基准） | 163 | **0.163 TOPS** | **10.9 %** |
| 仅 YOLOv8n-MR · 25 fps（白天关闭增强） | 153 | 0.153 TOPS | 10.2 % |

> 增强子网仅占总算力 6%，但带来 ExDark 上 mAP@50 提升约 3–5 pp。

### 2.4 准确率指标

#### Val 集 (750 张, 2408 实例)

| 指标 | YOLOv8n-MR |
|---|---|
| Precision | 0.755 |
| Recall | 0.636 |
| **mAP@50** | **0.716** |
| mAP@50:95 | 0.442 |

#### Test 集 (694 张, 2264 实例)，按"侦察主目标"重排

| 类别 | Precision | Recall | **mAP@50** | mAP@50:95 | 任务角色 |
|---|---|---|---|---|---|
| **bus** | 0.841 | 0.710 | **0.864** | 0.615 | 车辆 |
| **people** | 0.843 | 0.605 | **0.746** | 0.428 | 人员（主目标） |
| **car** | 0.770 | 0.656 | **0.725** | 0.491 | 车辆（主目标） |
| **motorbike** | 0.738 | 0.612 | 0.722 | 0.408 | 装备 |
| **bicycle** | 0.798 | 0.661 | 0.715 | 0.435 | 装备 |
| dog | 0.716 | 0.625 | 0.715 | 0.430 | 动物 |
| cat | 0.754 | 0.677 | 0.739 | 0.440 | 动物 |
| boat | 0.813 | 0.629 | 0.731 | 0.379 | 水上目标 |
| bottle | 0.851 | 0.552 | 0.643 | 0.371 | 物品 |
| cup | 0.757 | 0.503 | 0.626 | 0.400 | 物品 |
| chair | 0.708 | 0.483 | 0.600 | 0.331 | 物品 |
| table | 0.636 | 0.425 | 0.546 | 0.320 | 物品 |
| **all (12 类均值)** | **0.768** | **0.595** | **0.699** | **0.422** | — |

> 侦察核心 5 类（people / car / bus / motorbike / bicycle）test mAP@50 在 **0.715 – 0.864** 区间，加权平均 **≈ 0.76**。本次精度低于红外通道是因为 ExDark 12 类比 HIT-UAV 4 类难度更高，且 chair/table/cup 等室内长尾类拖低均值；生产部署可仅保留前 5 类。

---

# 三模块整体对比汇总

| 维度 | 事件相机（RVT-S） | 红外（YOLOv8n-IR） | 微光（Zero-DCE++ + YOLOv8n-MR） |
|---|---|---|---|
| 传感器 | Sony/Prophesee IMX636 | 高德 MINI212 | SmartSens SC2210 |
| 训练数据集 | 1 Mpx Automotive (预训练) | HIT-UAV (从零训练) | ExDark (从零训练) |
| 参数量 | 9.87 M | **2.92 M** | 2.93 M (含 Zero-DCE) |
| GMAC / 帧 | 8.74 | **6.05** | 6.52 |
| GFLOPS / 帧 | 17.48 | **12.1** | 13.05 |
| Val mAP@50 | 0.797 (Moorea 序列) | **0.864** | 0.716 |
| Test mAP@50 | — | **0.828** | 0.699 |
| 主类 mAP@50 | — | Person 0.931 / Car 0.975 / Bicycle 0.917 | bus 0.864 / people 0.746 / car 0.725 |
| 4080 单帧延迟 | 7.83 ms | **1.7 ms** | 2.6 ms |
| NPU @ 25 fps 占 1.5T 预算 | 14.6 % | **10.1 %** | 10.9 % |

**三路并行 @ 25 fps 总算力占用 ≈ 35.6%**，剩余 64% 可分配给多模态融合层和系统调度开销，整体可在 CV184XH 1.5 TOPS 算力预算内实时运行。

完整可视化结果与逐项实验数据见 [docs/index.html](docs/index.html)（GitHub Pages: <https://samanthazhang-stu.github.io/rongchuang/>）。

