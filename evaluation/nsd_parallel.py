# -*- coding: utf-8 -*-
# Parallel-by-case version of abdomen_NSD_Eval.py.
# Per-case/per-organ computation is byte-for-byte identical to the original
# (same tolerances, same Aorta/IVC/Esophagus z-cropping, same SurfaceDice calls);
# only the 50 cases are distributed across worker processes.
import os, sys, argparse
import numpy as np
import nibabel as nb
import pandas as pd
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance

join = os.path.join
basename = os.path.basename

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='')
parser.add_argument('--seg_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--workers', type=int, default=6)
args = parser.parse_args()

label_tolerance = OrderedDict({'Liver': 5, 'RK': 3, 'Spleen': 3, 'Pancreas': 5,
                               'Aorta': 2, 'IVC': 2, 'RAG': 2, 'LAG': 2, 'Gallbladder': 2,
                               'Esophagus': 3, 'Stomach': 5, 'Duodenum': 7, 'LK': 3})
ORGANS = list(label_tolerance.keys())


def find_lower_upper_zbound(organ_mask):
    organ_mask = np.uint8(organ_mask)
    z_index = np.where(organ_mask > 0)[2]
    return np.min(z_index), np.max(z_index)


def work(task):
    gt_path, seg_path, name = task
    gt_nii = nb.load(join(gt_path, name))
    case_spacing = gt_nii.header.get_zooms()
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())
    out = {'Name': name}
    for i, organ in enumerate(ORGANS, 1):
        e = gt_data == i
        p = seg_data == i
        if np.sum(e) == 0 and np.sum(p) == 0:
            nsd = 1
        elif np.sum(e) == 0 and np.sum(p) > 0:
            nsd = 0
        else:
            if i == 5 or i == 6 or i == 10:
                z_lower, z_upper = find_lower_upper_zbound(e)
                organ_gt = gt_data[:, :, z_lower:z_upper] == i
                organ_seg = seg_data[:, :, z_lower:z_upper] == i
            else:
                organ_gt, organ_seg = e, p
            sd = compute_surface_distances(organ_gt, organ_seg, case_spacing)
            nsd = compute_surface_dice_at_tolerance(sd, label_tolerance[organ])
        out['{}_NSD'.format(organ)] = round(nsd, 4)
    del gt_data, seg_data
    return out


if __name__ == '__main__':
    filenames = os.listdir(args.seg_path)
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    filenames = [x for x in filenames if os.path.exists(join(args.seg_path, x))]
    filenames.sort()
    tasks = [(args.gt_path, args.seg_path, n) for n in filenames]

    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            results[r['Name']] = r
            done += 1
            print('done %d/%d (%s)' % (done, len(tasks), r['Name']), flush=True)

    ordered = [results[n] for n in filenames]
    seg_metrics = OrderedDict()
    seg_metrics['Name'] = [r['Name'] for r in ordered]
    for organ in ORGANS:
        seg_metrics['{}_NSD'.format(organ)] = [r['{}_NSD'.format(organ)] for r in ordered]
    df = pd.DataFrame(seg_metrics)
    df.to_csv(args.save_path, index=False)

    avg = df.mean(axis=0, numeric_only=True).mean()
    print(20 * '>')
    print('Average NSD for %s: %s' % (basename(args.seg_path), avg) , flush=True)
    print(20 * '<')
    for organ in ORGANS:
        print('%-16s %.4f' % (organ, df['{}_NSD'.format(organ)].mean()), flush=True)
