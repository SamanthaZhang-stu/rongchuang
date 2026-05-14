"""
Run RVT-S on a STATIC-camera event recording (RAW EVT3 from Prophesee public demos).

Why this matters:
  RVT-S was trained on car-front-facing 1Mpx Automotive data (camera moves).
  Project use case is unmanned reconnaissance: camera is STATIC, targets move.
  This script feeds RVT-S a static-camera recording and writes a video,
  letting us judge whether the model generalizes to the target deployment scene.

Input: Prophesee RAW (EVT3) decoded by `expelliarmus`.
Pipeline:
  1. Decode events (x, y, p, t) from RAW with expelliarmus.
  2. Downsample-by-2 in space (x/2, y/2)  ->  640x360 plane.
  3. Cut into 50-ms windows, build StackedHistogram (20 ch, 360x640, uint8).
  4. Top-pad to (20, 384, 640) to match RVT's multiple-of-64 requirement.
  5. Streamed inference with hidden state carried across steps.
  6. Render polarity heatmap + detection overlays into an MP4.
"""
import os, sys, argparse, time, math
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

import torch
import cv2
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from expelliarmus import Wizard

from config.modifier import dynamically_modify_train_config
from modules.detection import Module as RnnDetModule
from data.utils.representations import StackedHistogram
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
    on  = t[:half].sum(axis=0).astype(np.float32)
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
    ap.add_argument('--raw', required=True, help='Prophesee RAW EVT3 file path')
    ap.add_argument('--encoding', default='EVT3', choices=['EVT2','EVT3','DAT'])
    ap.add_argument('--out_mp4', required=True)
    ap.add_argument('--scene_label', default='static camera scene')
    ap.add_argument('--max_seconds', type=float, default=999.0)
    ap.add_argument('--dt_us', type=int, default=50_000)
    ap.add_argument('--score_thr', type=float, default=0.3)
    ap.add_argument('--conf_thr', type=float, default=0.1)
    ap.add_argument('--nms_thr', type=float, default=0.45)
    ap.add_argument('--fps_out', type=int, default=20)
    ap.add_argument('--upscale', type=int, default=2)
    ap.add_argument('--downsample_by_2', type=int, default=1,
                    help='1 = halve x,y to fit 640x360 (use for native 1280x720 IMX636); '
                         '0 = keep raw resolution (use for VGA Gen3 input where x,y already <=640).')
    ap.add_argument('--evdense_min', type=float, default=0.0,
                    help='Drop detections whose bbox has < this fraction of pixels with at least one event. '
                         '0.05-0.15 effectively kills static-scene training-prior phantoms.')
    args = ap.parse_args()

    device = torch.device('cuda:0')
    out_mp4 = Path(args.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading RVT-{args.model} from {args.ckpt}')
    module, cfg = load_rvt(args.model, args.ckpt, device)
    in_hw = tuple(cfg.model.backbone.in_res_hw)   # (384, 640)
    in_channels = cfg.model.backbone.input_channels  # 20
    num_classes = cfg.model.head.num_classes  # 3
    bins = in_channels // 2  # 10

    pad_top = in_hw[0] - 360  # 24
    H_model, W_model = in_hw
    H_ds, W_ds = 360, 640

    # Decode RAW
    print(f'Decoding {args.raw} (encoding={args.encoding}) ...')
    wiz = Wizard(encoding=args.encoding, fpath=args.raw)
    events = wiz.read()
    print(f'  decoded {len(events):,} events; dtype={events.dtype}')
    if 't' in events.dtype.names: t_arr = events['t']
    else: t_arr = events[events.dtype.names[0]]
    x_arr = events['x']; y_arr = events['y']; p_arr = events['p']

    # Determine native resolution. Prophesee RAW usually 1280x720 for Gen4 / IMX636.
    src_w = int(x_arr.max()) + 1
    src_h = int(y_arr.max()) + 1
    print(f'  observed event range: x in [0, {src_w-1}], y in [0, {src_h-1}]')

    # Downsample-by-2 in space (or not, depending on flag)
    if args.downsample_by_2:
        x_ds = (x_arr.astype(np.int32)) // 2
        y_ds = (y_arr.astype(np.int32)) // 2
        print(f'  ds-by-2 applied: x,y now in [0, {x_ds.max()}], [0, {y_ds.max()}]')
    else:
        x_ds = x_arr.astype(np.int32)
        y_ds = y_arr.astype(np.int32)
        print(f'  ds-by-2 SKIPPED (raw resolution kept): x,y in [0, {x_ds.max()}], [0, {y_ds.max()}]')
    # Clip to model plane (360, 640) — events outside the IMX636-equivalent FOV are dropped
    in_fov = (x_ds < W_ds) & (y_ds < H_ds)
    x_ds = x_ds[in_fov]
    y_ds = y_ds[in_fov]
    dropped = (~in_fov).sum()
    if dropped > 0:
        print(f'  dropped {dropped:,} events outside {W_ds}x{H_ds} model FOV')
    p_full = p_arr.astype(np.int32)
    if p_full.min() < 0:
        p_full = (p_full + 1) // 2  # -1/+1 -> 0/1
    p_full = np.clip(p_full, 0, 1)
    p_ds = p_full[in_fov] if 'in_fov' in dir() else p_full
    t_us = (t_arr.astype(np.int64))[in_fov] if 'in_fov' in dir() else t_arr.astype(np.int64)
    # Sort by timestamp to fix non-monotonic warnings from expelliarmus
    if not np.all(np.diff(t_us) >= 0):
        order = np.argsort(t_us, kind='stable')
        t_us = t_us[order]; x_ds = x_ds[order]; y_ds = y_ds[order]; p_ds = p_ds[order]
        print('  sorted events by timestamp (non-monotonic detected)')

    # Stream slicing
    t0 = int(t_us[0])
    t_end = min(int(t_us[-1]), int(t_us[0] + args.max_seconds * 1e6))
    n_steps = int((t_end - t0) // args.dt_us)
    print(f'  duration to process: {(t_end-t0)/1e6:.1f}s -> {n_steps} steps @ {args.dt_us/1000:.0f}ms')

    repr_builder = StackedHistogram(bins=bins, height=H_ds, width=W_ds)

    up = args.upscale
    out_w, out_h = W_model * up, H_model * up
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_mp4), fourcc, args.fps_out, (out_w, out_h))
    assert writer.isOpened(), f'VideoWriter failed for {out_mp4}'

    states = None
    inf_times = []
    n_dets_total = 0
    cur_t = t0
    cursor = 0  # index into events stream

    with torch.inference_mode():
        for step in range(n_steps):
            t_lo, t_hi = cur_t, cur_t + args.dt_us
            cur_t = t_hi
            # advance cursor to t_lo
            while cursor < len(t_us) and t_us[cursor] < t_lo:
                cursor += 1
            # find end of window
            end = cursor
            while end < len(t_us) and t_us[end] < t_hi:
                end += 1
            if end == cursor:
                # empty window: feed zeros so the RNN clock keeps moving
                ev_tensor = torch.zeros((in_channels, H_ds, W_ds), dtype=torch.uint8)
            else:
                ev_tensor = repr_builder.construct(
                    torch.from_numpy(x_ds[cursor:end]),
                    torch.from_numpy(y_ds[cursor:end]),
                    torch.from_numpy(p_ds[cursor:end]),
                    torch.from_numpy(t_us[cursor:end]),
                )  # (20, 360, 640) uint8
            cursor = end

            ev_pad = torch.nn.functional.pad(ev_tensor.float(), (0, 0, pad_top, 0))
            inp = ev_pad.unsqueeze(0).to(device)

            t_start = time.perf_counter()
            outputs, _, states = module.mdl(inp, previous_states=states)
            torch.cuda.synchronize(device)
            inf_times.append(time.perf_counter() - t_start)

            processed = postprocess(prediction=outputs,
                                    num_classes=num_classes,
                                    conf_thre=args.conf_thr,
                                    nms_thre=args.nms_thr,
                                    class_agnostic=False)
            dets = processed[0]

            # Event-density gate: drop detections whose bbox interior has too few events.
            # This kills the "training-prior phantom" detections in static-camera scenes
            # where RVT-S hallucinates pedestrians/cars at positions it learned from the
            # car-front-facing 1Mpx training set.
            if dets is not None and dets.shape[0] > 0 and args.evdense_min > 0:
                # event activity per pixel = sum over the 20-channel stacked histogram
                act = ev_tensor.float().sum(dim=0).numpy()  # (H_ds, W_ds)
                # pad to model space (H_model, W_model)
                act_pad = np.pad(act, ((pad_top, 0), (0, 0)), mode='constant')
                d = dets.cpu().numpy()
                keep = []
                for k in range(d.shape[0]):
                    x1, y1, x2, y2 = int(max(0, d[k, 0])), int(max(0, d[k, 1])), \
                                     int(min(W_model, d[k, 2])), int(min(H_model, d[k, 3]))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    box_act = act_pad[y1:y2, x1:x2]
                    px_with_evt = (box_act > 0).sum()
                    total_px = box_act.size
                    if total_px == 0: continue
                    frac = px_with_evt / total_px
                    if frac >= args.evdense_min:
                        keep.append(k)
                dets = dets[keep] if keep else None

            if dets is not None:
                n_dets_total += int((dets[:, 4] * dets[:, 5] >= args.score_thr).sum())

            vis = make_event_frame(ev_tensor)
            vis = cv2.copyMakeBorder(vis, pad_top, 0, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(28, 28, 28))
            if dets is not None:
                vis = draw_dets(vis, dets.cpu().numpy(), args.score_thr)

            cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
            hdr = f'RVT-{args.model.upper()} 1Mpx | {args.scene_label} | t={ (cur_t - t0)/1e6:5.2f}s | step {step+1}/{n_steps}'
            cv2.putText(vis, hdr, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            n_dets_here = 0 if dets is None else int((dets[:, 4] * dets[:, 5] >= args.score_thr).sum())
            cv2.rectangle(vis, (0, vis.shape[0] - 22), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
            cv2.putText(vis,
                        f'STATIC CAMERA | sensor: IMX636 1280x720 (from {args.encoding}) | dets>={args.score_thr}: {n_dets_here}',
                        (4, vis.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            if up != 1:
                vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            writer.write(vis)

            if (step + 1) % 100 == 0:
                print(f'  [{step+1}/{n_steps}] avg lat={np.mean(inf_times)*1000:.2f}ms, total_dets={n_dets_total}')

    writer.release()
    print(f'\nDone. wrote {n_steps} frames -> {out_mp4}')
    print(f'  size: {out_mp4.stat().st_size/1024/1024:.1f} MB')
    print(f'  duration: {n_steps/args.fps_out:.1f}s @ {args.fps_out}fps')
    print(f'  inference mean: {np.mean(inf_times)*1000:.2f} ms/frame  ({1/np.mean(inf_times):.1f} fps)')
    print(f'  total detections (score>={args.score_thr}): {n_dets_total}')


if __name__ == '__main__':
    main()
