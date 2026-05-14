"""
Run RVT-S (1Mpx pretrained) on a Prophesee public Gen4.1 HDF5 sample
(IMX636-equivalent 1280x720 sensor) and save detection result images.

Pipeline:
  1. Read event stream from Prophesee HDF5 (EVT3-decoded x,y,p,t).
  2. Accumulate into stacked_histogram with dt=50ms, nbins=10 -> 20 channels.
  3. Downsample-by-2 (nearest, as RVT does at training time)
     -> tensor shape (20, 360, 640) per time step.
  4. Pad to (20, 384, 640) (multiple of 64).
  5. Sequentially feed to RVT-S with hidden state carried over time.
  6. Post-process YoloX outputs, draw bboxes on the event frame, save PNG.
"""
import os, sys, argparse, time, math
from pathlib import Path

PROJ = Path('/home/samantha_zhang/event_camera_imx636_yolo')
RVT = PROJ / 'RVT'
sys.path.insert(0, str(RVT))
os.chdir(RVT)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'

import numpy as np
import torch
import h5py
import cv2
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from config.modifier import dynamically_modify_train_config
from modules.detection import Module as RnnDetModule
from data.utils.representations import StackedHistogram
from models.detection.yolox.utils.boxes import postprocess

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# 1Mpx class names (RVT trained on 1Mpx detects these three classes)
CLASS_NAMES = ['pedestrian', 'two-wheeler', 'car']
CLASS_COLORS = [(0, 255, 0), (255, 255, 0), (0, 128, 255)]  # BGR


def load_rvt(mdl_cfg_name: str, ckpt_path: str, device):
    with initialize_config_dir(version_base='1.2', config_dir=str(RVT / 'config')):
        cfg = compose(config_name='val',
                      overrides=[
                          'dataset=gen4',
                          'dataset.path=/tmp/dummy',
                          f'checkpoint={ckpt_path}',
                          'use_test_set=0',
                          'hardware.gpus=0',
                          f'+experiment/gen4={mdl_cfg_name}.yaml',
                          'batch_size.eval=1',
                          'model.postprocess.confidence_threshold=0.1',
                      ])
    dynamically_modify_train_config(cfg)
    module = RnnDetModule.load_from_checkpoint(ckpt_path, full_config=cfg, strict=True)
    module.to(device).eval()
    return module, cfg


def read_prophesee_hdf5(path: Path):
    """Read x, y, p, t arrays from a Prophesee HDF5 dump.

    HDF5 produced by Prophesee Metavision SDK is normally laid out as:
      /CD/events   structured dtype [('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')]
    But export tooling differs; fall back to scanning datasets.
    """
    with h5py.File(str(path), 'r') as f:
        print(f'HDF5 root keys: {list(f.keys())}')
        # Try canonical layout
        if 'CD' in f and 'events' in f['CD']:
            ev = f['CD']['events'][...]
        elif 'events' in f:
            ev = f['events'][...]
        else:
            # try first big structured dataset
            found = None
            def visit(name, obj):
                nonlocal found
                if isinstance(obj, h5py.Dataset) and obj.dtype.names is not None and found is None:
                    found = (name, obj[...])
            f.visititems(visit)
            if found is None:
                raise RuntimeError(f'no event dataset found in {path}')
            print(f'fallback dataset: {found[0]}')
            ev = found[1]
        print(f'events dtype: {ev.dtype}, count: {len(ev):_}')
        names = ev.dtype.names
        x = ev[names[0]].astype(np.int32)
        y = ev[names[1]].astype(np.int32)
        pol = ev[names[2]].astype(np.int32)
        t = ev[names[3]].astype(np.int64)
    pol = np.clip(pol, 0, 1)  # convert -1/1 -> 0/1 if needed
    return x, y, pol, t


def event_chunks(x, y, pol, t, dt_us: int = 50_000):
    """Yield successive [t0, t0+dt) chunks of events sorted by time."""
    t0 = t[0]
    end = t[-1]
    start_idx = 0
    cur = t0
    while cur < end:
        cur_end = cur + dt_us
        # search end index
        idx_end = np.searchsorted(t, cur_end, side='left')
        yield x[start_idx:idx_end], y[start_idx:idx_end], pol[start_idx:idx_end], t[start_idx:idx_end], cur, cur_end
        start_idx = idx_end
        cur = cur_end


def render_event_frame(x, y, pol, H, W):
    """Render a polarity-coloured event frame for visualisation (BGR)."""
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    if len(x):
        # downsample bool mask to fit image, with x in [0, W*2), y in [0, H*2) (we already downsampled)
        ix = np.clip(x, 0, W - 1).astype(np.int32)
        iy = np.clip(y, 0, H - 1).astype(np.int32)
        # ON = bluish, OFF = reddish
        on = pol == 1
        off = ~on
        img[iy[off], ix[off]] = (60, 60, 220)   # red
        img[iy[on],  ix[on]]  = (220, 130, 60)  # blue/cyan
    return img


def draw_dets(img, dets, score_thr=0.3):
    if dets is None:
        return img
    dets = dets.cpu().numpy()
    for det in dets:
        x1, y1, x2, y2, obj_conf, cls_conf, cls = det
        score = obj_conf * cls_conf
        if score < score_thr: continue
        cls = int(cls)
        color = CLASS_COLORS[cls % 3]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{CLASS_NAMES[cls]} {score:.2f}'
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th_ - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', default='small', choices=['tiny','small','base'])
    ap.add_argument('--hdf5', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--name', default='sample')
    ap.add_argument('--max_chunks', type=int, default=80,
                    help='Stop after N 50-ms chunks (defaults ~4s).')
    ap.add_argument('--save_every', type=int, default=8,
                    help='Save 1 figure every K chunks.')
    ap.add_argument('--score_thr', type=float, default=0.3)
    ap.add_argument('--conf_thr', type=float, default=0.1)
    ap.add_argument('--nms_thr', type=float, default=0.45)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')

    print(f'Loading RVT-{args.model} from {args.ckpt}')
    module, cfg = load_rvt(args.model, args.ckpt, device)

    in_hw = tuple(cfg.model.backbone.in_res_hw)  # (384, 640) — model expects this
    in_channels = cfg.model.backbone.input_channels  # 20
    num_classes = cfg.model.head.num_classes  # 3
    bins = in_channels // 2  # 10

    # Source resolution (IMX636 / Gen4.1): 1280x720.
    src_h, src_w = 720, 1280
    # After downsample-by-2: 360x640. Then padded top to 384x640.
    after_ds = (src_h // 2, src_w // 2)
    pad_top = in_hw[0] - after_ds[0]  # 384 - 360 = 24
    pad_bot = 0
    print(f'IMX636 native: {src_w}x{src_h} | model in: {in_hw} | pad_top={pad_top}')

    repr_builder = StackedHistogram(bins=bins, height=after_ds[0], width=after_ds[1])

    print(f'Reading events from {args.hdf5}')
    x_all, y_all, p_all, t_all = read_prophesee_hdf5(Path(args.hdf5))
    # downsample-by-2: discard the LSB of x,y (RVT uses 'nearest' downsample)
    x_ds = x_all // 2
    y_ds = y_all // 2

    states = None
    saved = 0
    chunk_count = 0
    start_time = time.perf_counter()
    n_dets = 0

    print('Running inference …')
    with torch.inference_mode():
        for x, y, p, t, t0, t1 in event_chunks(x_ds, y_ds, p_all, t_all, dt_us=50_000):
            if chunk_count >= args.max_chunks: break
            chunk_count += 1

            if len(x) == 0:
                # no events: feed zeros so RNN state keeps advancing
                tensor_2D = torch.zeros((in_channels, after_ds[0], after_ds[1]), dtype=torch.uint8)
            else:
                tensor_2D = repr_builder.construct(
                    th.tensor := torch.from_numpy(x.astype(np.int32)),
                    torch.from_numpy(y.astype(np.int32)),
                    torch.from_numpy(p.astype(np.int32)),
                    torch.from_numpy(t.astype(np.int64)),
                )  # (20, 360, 640) uint8

            # pad top to (20, 384, 640)
            tensor = torch.nn.functional.pad(tensor_2D.float(), (0, 0, pad_top, pad_bot))
            tensor = tensor.unsqueeze(0).to(device)  # (1, 20, 384, 640)

            outputs, _, states = module.mdl(tensor, previous_states=states)

            # outputs is YoloX style: list/tensor at multiple strides. RVT's yolox_head
            # in eval mode returns concatenated tensor. Use postprocess.
            processed = postprocess(prediction=outputs,
                                    num_classes=num_classes,
                                    conf_thre=args.conf_thr,
                                    nms_thre=args.nms_thr,
                                    class_agnostic=False)
            dets = processed[0] if processed is not None else None
            if dets is not None:
                n_dets += int((dets[:, 4] * dets[:, 5] >= args.score_thr).sum())

            if chunk_count % args.save_every == 0 or chunk_count == args.max_chunks:
                # build visualisation frame at model resolution
                vis = render_event_frame(x, y, p, after_ds[0], after_ds[1])  # 360x640
                # apply same pad as model input so bbox coords match
                vis_pad = cv2.copyMakeBorder(vis, pad_top, pad_bot, 0, 0,
                                             cv2.BORDER_CONSTANT, value=(30, 30, 30))
                vis_pad = draw_dets(vis_pad, dets, score_thr=args.score_thr)
                # add header
                cv2.rectangle(vis_pad, (0, 0), (vis_pad.shape[1], 22), (0, 0, 0), -1)
                hdr = f'RVT-{args.model.upper()} 1Mpx | {args.name} | t={t0/1e6:.2f}s..{t1/1e6:.2f}s | chunk #{chunk_count}'
                cv2.putText(vis_pad, hdr, (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                fp = out_dir / f'{args.name}_rvt-{args.model}_chunk{chunk_count:04d}.png'
                cv2.imwrite(str(fp), vis_pad)
                saved += 1
                print(f'  chunk {chunk_count}: t={t0/1e6:.2f}s, evt={len(x):_}, dets={len(dets) if dets is not None else 0}, '
                      f'saved -> {fp.name}')

    total_time = time.perf_counter() - start_time
    fps = chunk_count / total_time
    print(f'\nDone. processed {chunk_count} chunks ({chunk_count * 50} ms of event data) in {total_time:.2f} s')
    print(f'Effective stream throughput: {fps:.1f} chunks/s (= {fps:.1f} fps for 50ms-window inference)')
    print(f'Saved {saved} figures to {out_dir}')
    print(f'Total detections kept (score>={args.score_thr}): {n_dets}')


if __name__ == '__main__':
    main()
