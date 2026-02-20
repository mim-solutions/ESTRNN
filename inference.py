import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from model import Model
from para import Parameter
from data.utils import normalize, normalize_reverse
from os.path import join, exists, isdir, dirname, basename


def print_timing_stats(name, times_s, n_output_frames):
    arr = np.array(times_s)
    total = arr.sum()
    print(f"\n{name} timing (s), n={len(arr)}:")
    print(f"  total={total:.4f}  per_output_frame={total / n_output_frames:.4f}")
    print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}")
    print(f"  min={arr.min():.4f}  p50={np.percentile(arr, 50):.4f}  "
          f"p90={np.percentile(arr, 90):.4f}  p99={np.percentile(arr, 99):.4f}  max={arr.max():.4f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help="the path of input video or video dir")
    parser.add_argument('--ckpt', type=str, required=True, help="the path of checkpoint of pretrained model")
    parser.add_argument('--dst', type=str, help="where to store the results")
    args = parser.parse_args()

    para = Parameter().args
    model = Model(para).cuda()
    checkpoint_path = args.ckpt
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])

    if not isdir(args.src):
        vid_cap = cv2.VideoCapture(args.src)
        num_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        args.src = join(dirname(args.src), basename(args.src).replace('.', '_'))
        os.makedirs(args.src, exist_ok=True)
        for i in range(num_frames):
            try:
                ret, img = vid_cap.read()
                cv2.imwrite(join(args.src, '{:08d}.png'.format(i)), img)
            except:
                break

    img_paths = sorted(os.listdir(args.src), key=lambda x: int(x.split('.')[0]))
    save_dir = args.dst
    if not exists(save_dir):
        os.makedirs(save_dir)
    seq_length = len(img_paths)
    if para.test_frames > seq_length:
        para.test_frames = seq_length
    start = 0
    end = para.test_frames
    val_range = 2.0 ** 8 - 1
    suffix = 'png'

    preprocess_times_s = []
    nn_times_s = []
    postprocess_times_s = []

    while True:
        input_seq = []
        for frame_idx in range(start, end):
            blur_img_path = join(args.src, img_paths[frame_idx])
            blur_img = cv2.imread(blur_img_path).transpose(2, 0, 1)[np.newaxis, ...]
            input_seq.append(blur_img)
        input_seq = np.concatenate(input_seq)[np.newaxis, :]

        model.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            input_seq = normalize(torch.from_numpy(input_seq).float().cuda(), centralize=para.centralize,
                                  normalize=para.normalize, val_range=val_range)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            preprocess_times_s.append(t1 - t0)

            t0 = time.perf_counter()
            output_seq = model([input_seq, ])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            nn_times_s.append(t1 - t0)

            if isinstance(output_seq, (list, tuple)):
                output_seq = output_seq[0]
            output_seq = output_seq.squeeze(dim=0)

        for frame_idx in range(para.past_frames, end - start - para.future_frames):
            blur_img = input_seq.squeeze(dim=0)[frame_idx]
            blur_img = normalize_reverse(blur_img, centralize=para.centralize, normalize=para.normalize,
                                         val_range=val_range)
            blur_img = blur_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
            blur_img = blur_img.astype(np.uint8)
            blur_img_path = join(save_dir, '{:08d}_input.{}'.format(frame_idx + start, suffix))

            t0 = time.perf_counter()
            deblur_img = output_seq[frame_idx - para.past_frames]
            deblur_img = normalize_reverse(deblur_img, centralize=para.centralize, normalize=para.normalize,
                                           val_range=val_range)
            deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
            deblur_img = np.clip(deblur_img, 0, val_range)
            deblur_img = deblur_img.astype(np.uint8)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            postprocess_times_s.append(t1 - t0)

            deblur_img_path = join(save_dir, '{:08d}_{}.{}'.format(frame_idx + start, para.model.lower(), suffix))
            cv2.imwrite(blur_img_path, blur_img)
            cv2.imwrite(deblur_img_path, deblur_img)

        if end == seq_length:
            break
        else:
            start = end - para.future_frames - para.past_frames
            end = start + para.test_frames
            if end > seq_length:
                end = seq_length
                start = end - para.test_frames

    n_output_frames = len(postprocess_times_s)
    print_timing_stats("Preprocessing  per window (numpy→tensor, CUDA transfer, normalize)", preprocess_times_s, n_output_frames)
    print_timing_stats("Neural network per window (forward pass)", nn_times_s, n_output_frames)
    print_timing_stats("Postprocessing per frame  (normalize_reverse, GPU→CPU, clip, cast)", postprocess_times_s, n_output_frames)
