# 传感器检测算法GPU验证

# 事件相机

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

