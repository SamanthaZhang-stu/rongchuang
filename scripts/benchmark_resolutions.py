"""
Sweep RVT-S FLOPS / latency over different input resolutions
to show the linear relationship FLOPS ~ H*W.
"""
import os, sys, time, math
from pathlib import Path

PROJ = Path('/home/samantha_zhang/event_camera_imx636_yolo')
RVT = PROJ / 'RVT'
sys.path.insert(0, str(RVT))
os.chdir(RVT)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from config.modifier import dynamically_modify_train_config
from modules.detection import Module as RnnDetModule
from fvcore.nn import FlopCountAnalysis

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# (label, source HxW from IMX636 stream, downsample-by-2 in pipeline?)
# After ceil-to-multiple-of-64 on dataloading HxW, you get mdl_hw.
CASES = [
    ('IMX636 native 1280x720, ds=2 (default in repo)',  720, 1280, True),
    ('IMX636 native 1280x720, ds=1 (full res)',         720, 1280, False),
    ('Half  640x360, ds=1',                              360,  640, False),
    ('Half  640x360, ds=2 (quarter pixels)',             360,  640, True),
    ('VGA-ish 512x256, ds=1',                            256,  512, False),
    ('Tiny  320x192, ds=1 (gen1-class)',                 192,  320, False),
]

def ceil64(x): return math.ceil(x / 64) * 64

def run_one(label, src_h, src_w, ds2, ckpt):
    if ds2:
        dl_h, dl_w = src_h // 2, src_w // 2
    else:
        dl_h, dl_w = src_h, src_w
    mdl_h, mdl_w = ceil64(dl_h), ceil64(dl_w)
    part_h, part_w = mdl_h // 64, mdl_w // 64
    if part_h == 0 or part_w == 0:
        print(f'  [skip] {label}: partition size too small')
        return None

    device = torch.device('cuda:0')
    with initialize_config_dir(version_base='1.2', config_dir=str(RVT/'config')):
        cfg = compose(config_name='val',
                      overrides=[
                          'dataset=gen4', 'dataset.path=/tmp/dummy',
                          f'checkpoint={ckpt}', 'use_test_set=0',
                          'hardware.gpus=0',
                          '+experiment/gen4=small.yaml',
                          'batch_size.eval=1',
                          f'dataset.downsample_by_factor_2={"True" if ds2 else "False"}',
                      ])
    # We must NOT call dynamically_modify_train_config under different dataset.HW
    # because get_dataloading_hw is hard-coded to gen4=720x1280 from the dataset name.
    # Instead, set HW manually:
    with open_dict(cfg):
        cfg.model.backbone.in_res_hw = (mdl_h, mdl_w)
        cfg.model.backbone.stage.attention.partition_size = (part_h, part_w)
        cfg.model.head.num_classes = 3

    module = RnnDetModule.load_from_checkpoint(ckpt, full_config=cfg, strict=True)
    module.to(device).eval()
    mdl = module.mdl

    C = cfg.model.backbone.input_channels
    x = torch.zeros((1, C, mdl_h, mdl_w), device=device)

    class W(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            f, _ = self.m.forward_backbone(x, previous_states=None)
            pf = self.m.fpn(f)
            o, _ = self.m.yolox_head(pf)
            return o

    with torch.inference_mode():
        fc = FlopCountAnalysis(W(mdl), x)
        fc.unsupported_ops_warnings(False); fc.uncalled_modules_warnings(False)
        n_mac = fc.total()
    gmac = n_mac / 1e9

    # measure latency
    states = None
    with torch.inference_mode():
        for _ in range(5):
            out, _, states = mdl(x, previous_states=states)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            out, _, states = mdl(x, previous_states=states)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / 30
    fps = 1.0 / lat

    npu_int8_25fps = gmac * 25 / 1000  # TOPS (1 MAC = 1 OP convention)
    pct = npu_int8_25fps / 1.5 * 100

    pixels = mdl_h * mdl_w
    print(f'\n--- {label} ---')
    print(f'  src HxW: {src_h}x{src_w},  ds_by_2={ds2}')
    print(f'  dataloading HxW (after ds): {dl_h}x{dl_w}')
    print(f'  model in HxW (ceil to 64):  {mdl_h}x{mdl_w}  partition={part_h}x{part_w}  pixels={pixels:,}')
    print(f'  GMAC/frame:                 {gmac:.3f}')
    print(f'  GMAC per Megapixel:         {gmac / (pixels/1e6):.3f}')
    print(f'  FP32 latency:               {lat*1000:.2f} ms ({fps:.0f} fps)')
    print(f'  NPU INT8 @25fps:            {npu_int8_25fps:.3f} TOPS ({pct:.1f}% of 1.5T)')

    del module, mdl
    torch.cuda.empty_cache()
    return {'label': label, 'mdl_h': mdl_h, 'mdl_w': mdl_w, 'pixels': pixels,
            'gmac': gmac, 'lat_ms': lat*1000, 'fps': fps,
            'tops_int8_25fps': npu_int8_25fps, 'pct_of_1p5T': pct}


if __name__ == '__main__':
    import json
    ckpt = '/home/samantha_zhang/event_camera_imx636_yolo/models/rvt-s-1mpx.ckpt'
    out = []
    for label, h, w, ds in CASES:
        r = run_one(label, h, w, ds, ckpt)
        if r: out.append(r)
    Path('/home/samantha_zhang/event_camera_imx636_yolo/results/metrics/resolution_sweep.json').write_text(
        json.dumps(out, indent=2))
    print('\n========= SUMMARY =========')
    print(f'{"Label":52s}{"pixels":>11s}{"GMAC":>8s}{"GMAC/MP":>9s}{"FPS":>7s}{"TOPS@25":>10s}{"%1.5T":>8s}')
    for r in out:
        print(f"{r['label'][:50]:52s}"
              f"{r['pixels']:>11,d}"
              f"{r['gmac']:>8.2f}"
              f"{r['gmac']/(r['pixels']/1e6):>9.2f}"
              f"{r['fps']:>7.0f}"
              f"{r['tops_int8_25fps']:>10.3f}"
              f"{r['pct_of_1p5T']:>7.1f}%")
