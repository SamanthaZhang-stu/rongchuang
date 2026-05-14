"""
Streamed inference + visualization on an RVT-format preprocessed sequence.

Uses the pre-existing stacked_histogram_dt=50_nbins=10 representation
(20-channel uint8 tensors at 360x640, ds-by-2 from IMX636 720x1280)
that ships inside the official RVT 1Mpx tar.

We extracted *one complete sequence* from the partial gen4.tar download:
  gen4/train/moorea_2019-06-14_002_976500000_1036500000/
This sequence was in the *training split* of RVT-S, so the mAP it produces
is upper-bound (the model has seen this clip). We use it primarily to
produce visualization images and confirm the inference pipeline.

Outputs:
  - results/figures/<seq>/*.png         detection overlays (every K frames)
  - results/metrics/<seq>_dets.json     raw box dump per frame
"""
import os, sys, time, json, argparse
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
import h5py
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from config.modifier import dynamically_modify_train_config
from modules.detection import Module as RnnDetModule
from models.detection.yolox.utils.boxes import postprocess
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import BBOX_DTYPE

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

CLASS_NAMES = ['pedestrian', 'two-wheeler', 'car']
CLASS_COLORS = [(60, 220, 60), (60, 220, 220), (60, 130, 220)]  # BGR


def load_rvt(mdl_cfg_name: str, ckpt_path: str, device):
    with initialize_config_dir(version_base='1.2', config_dir=str(RVT / 'config')):
        cfg = compose(config_name='val',
                      overrides=[
                          'dataset=gen4',
                          f'dataset.path=/tmp/dummy',  # not used for model build
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


def make_event_frame(ev_tensor_chw):
    """Render an event activity heatmap from the 20-channel stacked-histogram tensor (uint8).

    Channels = [ON polarity x 10 time-bins, OFF polarity x 10 time-bins]."""
    t = ev_tensor_chw.numpy() if hasattr(ev_tensor_chw, 'numpy') else ev_tensor_chw
    C, H, W = t.shape
    half = C // 2
    on = t[:half].sum(axis=0).astype(np.float32)     # (H, W)
    off = t[half:].sum(axis=0).astype(np.float32)    # (H, W)
    # normalize to [0, 1] using a robust percentile
    def norm(x):
        v = np.percentile(x[x > 0], 98) if (x > 0).any() else 1.0
        return np.clip(x / max(v, 1.0), 0, 1)
    on = norm(on); off = norm(off)
    img = np.full((H, W, 3), 28, dtype=np.uint8)
    img[..., 0] = np.maximum(img[..., 0], (on * 255).astype(np.uint8))           # blue from ON
    img[..., 1] = np.maximum(img[..., 1], (np.clip(on, 0, 1) * 80).astype(np.uint8))
    img[..., 2] = np.maximum(img[..., 2], (off * 255).astype(np.uint8))          # red from OFF
    return img


def draw_dets(img, dets_xyxy, score_thr=0.3):
    if dets_xyxy is None: return img
    for det in dets_xyxy:
        x1, y1, x2, y2, obj, cls_conf, cls = det
        score = float(obj) * float(cls_conf)
        if score < score_thr: continue
        cls = int(cls)
        if cls < 0 or cls >= len(CLASS_NAMES): continue
        color = CLASS_COLORS[cls]
        x1, y1, x2, y2 = (int(max(0, x1)), int(max(0, y1)),
                          int(max(0, x2)), int(max(0, y2)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{CLASS_NAMES[cls]} {score:.2f}'
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(0, y1 - th_ - 4)
        cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_gt(img, gt_xyxy_cls, downsample=2):
    if gt_xyxy_cls is None or len(gt_xyxy_cls) == 0:
        return img
    for x1, y1, x2, y2, cls in gt_xyxy_cls:
        x1, y1, x2, y2 = (int(max(0, x1)), int(max(0, y1)),
                          int(max(0, x2)), int(max(0, y2)))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        label = f'GT:{CLASS_NAMES[int(cls)]}'
        cv2.putText(img, label, (x1, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', default='small', choices=['tiny','small','base'])
    ap.add_argument('--seq_dir', required=True,
                    help='Directory containing event_representations_v2/ and labels_v2/')
    ap.add_argument('--out_root', default=str(PROJ / 'results'))
    ap.add_argument('--max_frames', type=int, default=400,
                    help='Stop after this many 50ms steps (default 400 = 20s).')
    ap.add_argument('--save_every', type=int, default=10)
    ap.add_argument('--score_thr', type=float, default=0.3)
    ap.add_argument('--conf_thr', type=float, default=0.1)
    ap.add_argument('--nms_thr', type=float, default=0.45)
    args = ap.parse_args()

    device = torch.device('cuda:0')
    seq_dir = Path(args.seq_dir)
    seq_name = seq_dir.name
    fig_dir = Path(args.out_root) / 'figures' / seq_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = Path(args.out_root) / 'metrics'
    metric_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading RVT-{args.model} from {args.ckpt}')
    module, cfg = load_rvt(args.model, args.ckpt, device)
    in_hw = tuple(cfg.model.backbone.in_res_hw)  # (384, 640)
    in_channels = cfg.model.backbone.input_channels  # 20
    num_classes = cfg.model.head.num_classes
    pad_top = in_hw[0] - 360  # 24

    # Load preprocessed data
    h5_path = seq_dir / 'event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5'
    ts_path = seq_dir / 'event_representations_v2/stacked_histogram_dt=50_nbins=10/timestamps_us.npy'
    objframe_to_repr_path = seq_dir / 'event_representations_v2/stacked_histogram_dt=50_nbins=10/objframe_idx_2_repr_idx.npy'
    labels_path = seq_dir / 'labels_v2/labels.npz'

    timestamps_us = np.load(str(ts_path))
    print(f'Sequence: {seq_name} | {timestamps_us.size} 50ms frames | duration {timestamps_us[-1]/1e6:.1f}s')

    objframe_to_repr_idx = np.load(str(objframe_to_repr_path))
    labels_data = np.load(str(labels_path))
    labels = labels_data['labels']                                 # structured array
    objframe_to_label_idx = labels_data['objframe_idx_2_label_idx']  # (N,) of int

    # Build repr_idx -> (gt boxes, class_ids) map (using downsample-by-2 transform: divide x,y,w,h by 2)
    repr_idx_to_gt = {}
    for ofi, repr_idx in enumerate(objframe_to_repr_idx):
        i0 = objframe_to_label_idx[ofi]
        i1 = objframe_to_label_idx[ofi + 1] if ofi + 1 < len(objframe_to_label_idx) else len(labels)
        if i0 == i1: continue
        boxes_for_frame = []
        for lab in labels[i0:i1]:
            x = lab['x'] / 2.0; y = lab['y'] / 2.0
            w = lab['w'] / 2.0; h = lab['h'] / 2.0
            # apply pad_top to y so frame coords match the padded image
            boxes_for_frame.append((x, y + pad_top, x + w, y + pad_top + h, int(lab['class_id'])))
        repr_idx_to_gt[int(repr_idx)] = boxes_for_frame

    # PropheseeEvaluator gives the same mAP that RVT validation reports
    evaluator = PropheseeEvaluator(dataset='gen4', downsample_by_2=True)

    states = None
    n_frames = min(args.max_frames, timestamps_us.size)
    print(f'Running inference on first {n_frames} 50ms steps …')

    saved = 0
    total_dets = 0
    inf_times = []
    dets_log = []

    with h5py.File(str(h5_path), 'r') as h5f:
        with torch.inference_mode():
            for i in range(n_frames):
                # load one 50ms frame (1, 20, 360, 640)
                ev = h5f['data'][i]  # (20, 360, 640) uint8
                ev_t = torch.from_numpy(ev).float()
                # pad to 384x640 (top pad 24)
                ev_pad = torch.nn.functional.pad(ev_t, (0, 0, pad_top, 0))
                inp = ev_pad.unsqueeze(0).to(device)
                # forward
                t0 = time.perf_counter()
                outputs, _, states = module.mdl(inp, previous_states=states)
                torch.cuda.synchronize(device)
                inf_times.append(time.perf_counter() - t0)
                # postprocess
                processed = postprocess(prediction=outputs,
                                        num_classes=num_classes,
                                        conf_thre=args.conf_thr,
                                        nms_thre=args.nms_thr,
                                        class_agnostic=False)
                dets = processed[0]  # (M, 7) or None
                # accumulate for mAP — only frames that have GT
                gt_for_frame = repr_idx_to_gt.get(i)
                if gt_for_frame is not None and len(gt_for_frame) > 0:
                    # build GT structured array (xywh, before pad)
                    gt_arr = np.zeros(len(gt_for_frame), dtype=BBOX_DTYPE)
                    for j, (x1, y1, x2, y2, c) in enumerate(gt_for_frame):
                        y1_un = y1 - pad_top
                        y2_un = y2 - pad_top
                        gt_arr[j] = (int(timestamps_us[i]), float(x1), float(y1_un),
                                     float(x2 - x1), float(y2_un - y1_un),
                                     int(c), 0, 1.0)
                    # build prediction structured array
                    if dets is not None and dets.shape[0] > 0:
                        d = dets.detach().cpu().numpy()
                        pred_arr = np.zeros(d.shape[0], dtype=BBOX_DTYPE)
                        pred_arr['t'] = int(timestamps_us[i])
                        pred_arr['x'] = d[:, 0]
                        pred_arr['y'] = (d[:, 1] - pad_top).clip(min=0)
                        pred_arr['w'] = d[:, 2] - d[:, 0]
                        pred_arr['h'] = (d[:, 3] - d[:, 1])
                        pred_arr['class_id'] = d[:, 6].astype(np.uint32)
                        pred_arr['class_confidence'] = (d[:, 4] * d[:, 5]).astype(np.float32)
                    else:
                        pred_arr = np.zeros(0, dtype=BBOX_DTYPE)
                    evaluator.add_labels([gt_arr])
                    evaluator.add_predictions([pred_arr])

                if dets is not None:
                    total_dets += int((dets[:, 4] * dets[:, 5] >= args.score_thr).sum())

                if i % args.save_every == 0 or i == n_frames - 1:
                    vis = make_event_frame(ev_t)  # 360x640
                    vis = cv2.copyMakeBorder(vis, pad_top, 0, 0, 0,
                                             cv2.BORDER_CONSTANT, value=(28, 28, 28))
                    # draw GT first (white thin), then dets (colored thick)
                    if i in repr_idx_to_gt:
                        vis = draw_gt(vis, repr_idx_to_gt[i])
                    if dets is not None:
                        vis = draw_dets(vis, dets.cpu().numpy(), score_thr=args.score_thr)
                    # header (two lines, top + bottom)
                    cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
                    hdr1 = f'RVT-{args.model.upper()} 1Mpx | t={timestamps_us[i]/1e6:5.2f}s | step {i+1}/{n_frames}'
                    cv2.putText(vis, hdr1, (4, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    n_dets_here = 0 if dets is None else int((dets[:, 4] * dets[:, 5] >= args.score_thr).sum())
                    # bottom bar
                    bot_y = vis.shape[0] - 16
                    cv2.rectangle(vis, (0, vis.shape[0] - 22), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
                    cv2.putText(vis, f'IMX636 1280x720 -> 640x384 | dets>={args.score_thr}: {n_dets_here}  GT (white): {len(repr_idx_to_gt.get(i,[]))}',
                                (4, bot_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                    fp = fig_dir / f'frame_{i:04d}.png'
                    cv2.imwrite(str(fp), vis)
                    saved += 1

                if dets is not None:
                    dets_log.append({
                        'frame': i,
                        't_us': int(timestamps_us[i]),
                        'boxes': [
                            {'xyxy': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                             'obj': float(b[4]), 'cls_conf': float(b[5]), 'cls': int(b[6])}
                            for b in dets.cpu().numpy()
                        ],
                    })

    mean_lat = float(np.mean(inf_times)) * 1000
    fps_calc = 1.0 / np.mean(inf_times)
    print(f'\n=== Inference summary ===')
    print(f'frames processed: {n_frames}')
    print(f'  mean latency: {mean_lat:.2f} ms/frame  →  {fps_calc:.1f} fps')
    print(f'  saved figures: {saved} -> {fig_dir}')
    print(f'  total dets (score>={args.score_thr}): {total_dets}')

    print(f'\n=== Computing mAP via Prophesee evaluator ===')
    try:
        metrics = evaluator.evaluate_buffer(img_height=in_hw[0] - pad_top, img_width=in_hw[1])
        print(json.dumps(metrics, indent=2, default=str))
    except Exception as e:
        print(f'evaluator error: {e}')
        metrics = {'error': str(e)}

    # write all dets to json
    with open(metric_dir / f'{seq_name}_dets.json', 'w') as f:
        json.dump({'seq': seq_name, 'n_frames': n_frames, 'detections': dets_log[:50],  # keep only 50 to limit size
                   'mean_latency_ms': mean_lat,
                   'fps': fps_calc,
                   'eval_metrics': metrics if isinstance(metrics, dict) else str(metrics)},
                  f, indent=2, default=str)
    print(f'Saved dets+metrics to {metric_dir / (seq_name + "_dets.json")}')


if __name__ == '__main__':
    main()
