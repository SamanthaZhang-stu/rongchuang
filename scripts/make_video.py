"""Run inference end-to-end and write an MP4 video of detection overlays.

Saves every frame in-memory and pipes to OpenCV's mp4v VideoWriter.
1198 frames at 20 fps -> ~60-second 1Mpx event-detection demo video.
"""
import os, sys, time, argparse
from pathlib import Path
import numpy as np

PROJ = Path('/home/samantha_zhang/event_camera_imx636_yolo')
RVT = PROJ / 'RVT'
sys.path.insert(0, str(RVT))
os.chdir(RVT)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ.setdefault(
    'HDF5_PLUGIN_PATH',
    '/home/samantha_zhang/mambaforge/envs/events_signals/lib/python3.11/site-packages/hdf5plugin/plugins'
)

import torch
import cv2
import h5py
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from config.modifier import dynamically_modify_train_config
from modules.detection import Module as RnnDetModule
from models.detection.yolox.utils.boxes import postprocess

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

CLASS_NAMES = ['pedestrian', 'two-wheeler', 'car']
CLASS_COLORS = [(60, 220, 60), (60, 220, 220), (60, 130, 220)]


def load_rvt(mdl_cfg_name, ckpt_path, device):
    with initialize_config_dir(version_base='1.2', config_dir=str(RVT / 'config')):
        cfg = compose(config_name='val',
                      overrides=[
                          'dataset=gen4', 'dataset.path=/tmp/dummy',
                          f'checkpoint={ckpt_path}', 'use_test_set=0',
                          'hardware.gpus=0',
                          f'+experiment/gen4={mdl_cfg_name}.yaml',
                          'batch_size.eval=1',
                          'model.postprocess.confidence_threshold=0.1',
                      ])
    dynamically_modify_train_config(cfg)
    module = RnnDetModule.load_from_checkpoint(ckpt_path, full_config=cfg, strict=True)
    module.to(device).eval()
    return module, cfg


def make_event_frame(ev_tensor):
    t = ev_tensor.numpy() if hasattr(ev_tensor, 'numpy') else ev_tensor
    C, H, W = t.shape
    half = C // 2
    on = t[:half].sum(axis=0).astype(np.float32)
    off = t[half:].sum(axis=0).astype(np.float32)
    def norm(x):
        v = np.percentile(x[x > 0], 98) if (x > 0).any() else 1.0
        return np.clip(x / max(v, 1.0), 0, 1)
    on = norm(on); off = norm(off)
    img = np.full((H, W, 3), 28, dtype=np.uint8)
    img[..., 0] = np.maximum(img[..., 0], (on * 255).astype(np.uint8))
    img[..., 1] = np.maximum(img[..., 1], (on * 80).astype(np.uint8))
    img[..., 2] = np.maximum(img[..., 2], (off * 255).astype(np.uint8))
    return img


def draw_dets(img, dets, score_thr):
    if dets is None: return img
    for det in dets:
        x1, y1, x2, y2, obj, cls_c, cls = det
        score = float(obj) * float(cls_c)
        if score < score_thr: continue
        cls = int(cls)
        if cls < 0 or cls >= len(CLASS_NAMES): continue
        color = CLASS_COLORS[cls]
        x1, y1, x2, y2 = (int(max(0, x1)), int(max(0, y1)),
                          int(max(0, x2)), int(max(0, y2)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lab = f'{CLASS_NAMES[cls]} {score:.2f}'
        (tw, th_), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th_ - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, lab, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', default='small', choices=['tiny','small','base'])
    ap.add_argument('--seq_dir', required=True)
    ap.add_argument('--out_mp4', required=True)
    ap.add_argument('--max_frames', type=int, default=1198)
    ap.add_argument('--score_thr', type=float, default=0.3)
    ap.add_argument('--conf_thr', type=float, default=0.1)
    ap.add_argument('--nms_thr', type=float, default=0.45)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--upscale', type=int, default=2,
                    help='Spatial upscale factor for nicer video (does NOT affect inference).')
    args = ap.parse_args()

    device = torch.device('cuda:0')
    seq_dir = Path(args.seq_dir)
    out_mp4 = Path(args.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading RVT-{args.model} from {args.ckpt}')
    module, cfg = load_rvt(args.model, args.ckpt, device)
    in_hw = tuple(cfg.model.backbone.in_res_hw)  # (384, 640)
    in_channels = cfg.model.backbone.input_channels  # 20
    num_classes = cfg.model.head.num_classes  # 3
    pad_top = in_hw[0] - 360

    h5_path = seq_dir / 'event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5'
    ts = np.load(str(seq_dir / 'event_representations_v2/stacked_histogram_dt=50_nbins=10/timestamps_us.npy'))
    n = min(args.max_frames, ts.size)
    print(f'Encoding {n} frames -> {out_mp4}')

    # video writer: (W*up, H*up) with mp4v
    up = args.upscale
    out_w, out_h = 640 * up, 384 * up
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_mp4), fourcc, args.fps, (out_w, out_h))
    assert writer.isOpened(), f'VideoWriter failed for {out_mp4}'

    states = None
    inf_times = []
    n_dets_total = 0

    with h5py.File(str(h5_path), 'r') as h5f:
        with torch.inference_mode():
            for i in range(n):
                ev = h5f['data'][i]
                ev_t = torch.from_numpy(ev).float()
                ev_pad = torch.nn.functional.pad(ev_t, (0, 0, pad_top, 0))
                inp = ev_pad.unsqueeze(0).to(device)

                t0 = time.perf_counter()
                outputs, _, states = module.mdl(inp, previous_states=states)
                torch.cuda.synchronize(device)
                inf_times.append(time.perf_counter() - t0)

                processed = postprocess(prediction=outputs,
                                        num_classes=num_classes,
                                        conf_thre=args.conf_thr,
                                        nms_thre=args.nms_thr,
                                        class_agnostic=False)
                dets = processed[0]
                if dets is not None:
                    n_dets_total += int((dets[:, 4] * dets[:, 5] >= args.score_thr).sum())

                vis = make_event_frame(ev_t)            # 360 x 640
                vis = cv2.copyMakeBorder(vis, pad_top, 0, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(28, 28, 28))
                if dets is not None:
                    vis = draw_dets(vis, dets.cpu().numpy(), args.score_thr)

                cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
                hdr = f'RVT-{args.model.upper()} 1Mpx | IMX636 1280x720 | t={ts[i]/1e6:5.2f}s | step {i+1}/{n}'
                cv2.putText(vis, hdr, (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                n_dets_here = 0 if dets is None else int((dets[:, 4] * dets[:, 5] >= args.score_thr).sum())
                cv2.rectangle(vis, (0, vis.shape[0] - 22), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
                cv2.putText(vis, f'detections (score>={args.score_thr}): {n_dets_here}',
                            (4, vis.shape[0] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                if up != 1:
                    vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                writer.write(vis)

                if (i + 1) % 100 == 0:
                    print(f'  [{i+1}/{n}] avg lat={np.mean(inf_times)*1000:.2f}ms, total_dets={n_dets_total}')

    writer.release()
    print(f'\nDone. wrote {n} frames -> {out_mp4}')
    print(f'  size: {out_mp4.stat().st_size/1024/1024:.1f} MB')
    print(f'  duration: {n/args.fps:.1f}s @ {args.fps}fps')
    print(f'  inference mean: {np.mean(inf_times)*1000:.2f} ms/frame  ({1/np.mean(inf_times):.1f} fps)')
    print(f'  total detections (score>={args.score_thr}): {n_dets_total}')


if __name__ == '__main__':
    main()
