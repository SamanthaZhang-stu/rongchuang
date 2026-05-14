"""
Benchmark RVT-S (1Mpx pretrained) on IMX636-equivalent resolution (1280x720).

Measures:
  - Parameter count (M)
  - FLOPs (GFLOPS) / MACs (GMAC) per single forward step
  - Peak VRAM (MB) during forward
  - Throughput / FPS at single-frame and sequence-length=5 inference
"""
import os, sys, time, json, gc
from pathlib import Path

PROJ = Path('/home/samantha_zhang/event_camera_imx636_yolo')
RVT = PROJ / 'RVT'
sys.path.insert(0, str(RVT))
os.chdir(RVT)

# RVT's validation.py thread settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'

import torch
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from config.modifier import dynamically_modify_train_config
from modules.detection import Module as RnnDetModule
from fvcore.nn import FlopCountAnalysis, parameter_count

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def build_module(mdl_cfg_name: str, ckpt_path: str) -> torch.nn.Module:
    """Initialize a RVT module + load checkpoint following validation.py logic."""
    with initialize_config_dir(version_base='1.2',
                               config_dir=str(RVT / 'config')):
        cfg = compose(config_name='val',
                      overrides=[
                          'dataset=gen4',
                          'dataset.path=/tmp/dummy',  # unused at model build time
                          f'checkpoint={ckpt_path}',
                          'use_test_set=0',
                          'hardware.gpus=0',
                          f'+experiment/gen4={mdl_cfg_name}.yaml',
                          'batch_size.eval=1',
                          'model.postprocess.confidence_threshold=0.1',
                      ])
    dynamically_modify_train_config(cfg)
    print("------ Resolved config (model) ------")
    print(OmegaConf.to_yaml(cfg.model))
    print(f"sequence_length={cfg.dataset.sequence_length}")
    module = RnnDetModule.load_from_checkpoint(ckpt_path, full_config=cfg, strict=True)
    module.eval()
    return module, cfg


def measure_params(model: torch.nn.Module) -> dict:
    counts = parameter_count(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params_M': total / 1e6,
        'trainable_params_M': trainable / 1e6,
        'backbone_params_M': counts.get('mdl.backbone', 0) / 1e6,
        'fpn_params_M': counts.get('mdl.fpn', 0) / 1e6,
        'head_params_M': counts.get('mdl.yolox_head', 0) / 1e6,
    }


def measure_flops_one_step(module, in_hw, in_channels, device):
    """Measure FLOPs for ONE single timestep through backbone + FPN + head."""
    mdl = module.mdl.to(device).eval()
    h, w = in_hw
    x = torch.zeros((1, in_channels, h, w), device=device)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            backbone_features, _ = self.m.forward_backbone(x, previous_states=None)
            # head + FPN
            fpn_features = self.m.fpn(backbone_features)
            out, _ = self.m.yolox_head(fpn_features)
            return out

    wrap = Wrapper(mdl)
    with torch.inference_mode():
        flops = FlopCountAnalysis(wrap, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        n_mac = flops.total()
    return {'gmac_per_step': n_mac / 1e9, 'gflops_per_step': n_mac * 2 / 1e9}


def measure_fps_vram(module, in_hw, in_channels, device, n_warmup=10, n_iter=50, batch=1):
    """Stream inference: feed time-step by time-step with hidden state. Measure VRAM + FPS."""
    mdl = module.mdl.to(device).eval()
    h, w = in_hw
    x = torch.randn((batch, in_channels, h, w), device=device)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device) / (1024**2)

    # warm-up with hidden state propagation
    states = None
    with torch.inference_mode():
        for _ in range(n_warmup):
            out, _, states = mdl(x, previous_states=states)
        torch.cuda.synchronize(device)

        # timed loop
        start = time.perf_counter()
        for _ in range(n_iter):
            out, _, states = mdl(x, previous_states=states)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    fps = n_iter * batch / elapsed
    latency_ms = elapsed / n_iter * 1000
    peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    resv = torch.cuda.max_memory_reserved(device) / (1024**2)
    return {
        'batch': batch,
        'latency_ms_per_step': latency_ms,
        'fps': fps,
        'mem_before_MB': mem_before,
        'peak_alloc_MB': peak,
        'peak_reserved_MB': resv,
    }


def main(mdl: str, ckpt: str, save_to: str):
    device = torch.device('cuda:0')
    print(f'\n=== RVT-{mdl.upper()} 1Mpx benchmark ===')
    print(f'ckpt: {ckpt}')

    module, cfg = build_module(mdl, ckpt)
    module = module.to(device).eval()

    in_hw = tuple(cfg.model.backbone.in_res_hw)
    in_channels = cfg.model.backbone.input_channels
    print(f'Effective input tensor shape (per step): (B={1}, C={in_channels}, H={in_hw[0]}, W={in_hw[1]})')
    print(f'Note: this maps from IMX636 native 1280x720 -> downsample x2 -> ceil to multiple-of-64 = {in_hw[1]}x{in_hw[0]}')

    params = measure_params(module)
    print(f'\nParameters:')
    for k, v in params.items():
        print(f'  {k}: {v:.3f}')

    flops = measure_flops_one_step(module, in_hw, in_channels, device)
    print(f'\nCompute (single forward step, B=1):')
    print(f'  GMAC: {flops["gmac_per_step"]:.3f}')
    print(f'  GFLOPS: {flops["gflops_per_step"]:.3f}')

    # FPS at B=1 (deployment scenario)
    perf_b1 = measure_fps_vram(module, in_hw, in_channels, device, batch=1)
    print(f'\nRuntime @ B=1 (FP32, RTX 4060 Laptop):')
    print(f'  Latency: {perf_b1["latency_ms_per_step"]:.2f} ms/step')
    print(f'  FPS: {perf_b1["fps"]:.1f}')
    print(f'  Peak VRAM (allocated): {perf_b1["peak_alloc_MB"]:.1f} MB')
    print(f'  Peak VRAM (reserved):  {perf_b1["peak_reserved_MB"]:.1f} MB')

    # FP16 throughput
    torch.cuda.empty_cache()
    module_fp16 = module.half()
    h, w = in_hw
    x16 = torch.randn((1, in_channels, h, w), device=device, dtype=torch.half)
    states = None
    with torch.inference_mode():
        for _ in range(10):
            out, _, states = module_fp16.mdl(x16, previous_states=states)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(50):
            out, _, states = module_fp16.mdl(x16, previous_states=states)
        torch.cuda.synchronize(device)
        elapsed16 = time.perf_counter() - start
    fps16 = 50 / elapsed16
    lat16 = elapsed16 / 50 * 1000
    print(f'\nRuntime @ B=1 (FP16):')
    print(f'  Latency: {lat16:.2f} ms/step  FPS: {fps16:.1f}')

    out = {
        'model': f'RVT-{mdl}',
        'dataset_pretrain': '1 Mpx Automotive (Prophesee Gen4)',
        'sensor_target': 'Sony/Prophesee IMX636 (1280x720)',
        'classes': ['pedestrian', 'two-wheeler', 'car'],
        'input_tensor_shape': {'C': in_channels, 'H': in_hw[0], 'W': in_hw[1],
                               'note': 'downsample_by_2 then ceil-to-multiple-of-64'},
        'params': params,
        'compute_per_step': flops,
        'runtime_fp32_b1': perf_b1,
        'runtime_fp16_b1': {'latency_ms_per_step': lat16, 'fps': fps16},
        'hardware': 'NVIDIA RTX 4060 Laptop GPU (8GB), CUDA 12.6',
        'note_npu_budget': f'Estimated INT8 NPU load @ 25 fps: GMAC * 1 * 25 = {flops["gmac_per_step"] * 25:.1f} GOPS/s ~= {flops["gmac_per_step"] * 25 / 1000:.3f} TOPS (INT8 budget on CV184XH = 1.5 TOPS)',
    }
    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    with open(save_to, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved metrics -> {save_to}')
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='small', choices=['tiny','small','base'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or f'/home/samantha_zhang/event_camera_imx636_yolo/results/metrics/rvt-{args.model}-benchmark.json'
    main(args.model, args.ckpt, out)
