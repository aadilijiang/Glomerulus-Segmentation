import os

import tifffile as tif
from PIL import Image
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, maybe_mkdir_p
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/data/zucksliu/Glom-Segmnentation/media/fabian/', help='Input dir path') 
    parser.add_argument('--raw_label_gt_path', type=str, default='data/raw/label_gt_FullStack', help='Raw label path') # data/raw/label_gt_FullStack_glom2
    parser.add_argument('--processed_tgt_label_path', type=str, default=f'data/processed/labelTr_new', help='Processed target label path') # data/processed/labelTs_new
    parser.add_argument('--date', type=str, default='04032024', help='Date')
    parser.add_argument('--threshold', type=int, default=250, help='Threshold')
    parser.add_argument('--file_suffix', type=str, default='.tif', help='File suffix')
    args = parser.parse_args()

    root_path = args.root_path

    raw_label_gt_path = args.root_path + args.raw_label_gt_path + f'_{args.date}/'
    processed_tgt_label_path = args.root_path + args.processed_tgt_label_path + f'_{args.date}/'

    maybe_mkdir_p(raw_label_gt_path)
    maybe_mkdir_p(processed_tgt_label_path)

    threshold = args.threshold

    images = subfiles(raw_label_gt_path, suffix=args.file_suffix, sort=True, join=False)

    print('Processing raw label data ...')
    for i, im in enumerate(images):
        print('Processing image', i, '/', len(images), ':', im)
        cur_tif = tif.imread(raw_label_gt_path + im)
        num_channels = cur_tif.shape[1]
        im_new = np.zeros_like(cur_tif[:, 0, :, :])

        #single_channel_tif = cur_tif[:, 3, :, :]
        #im_new[single_channel_tif > threshold] = 5 # Bowman

        #single_channel_tif = cur_tif[:, 0, :, :]
        #im_new[single_channel_tif > threshold] = 4 # Blood

        single_channel_tif = cur_tif[:, 2, :, :]
        im_new[single_channel_tif > threshold] = 3 # Capsule

        single_channel_tif = cur_tif[:, 1, :, :]
        im_new[single_channel_tif > threshold] = 2 # GBM

        single_channel_tif = cur_tif[:, 4, :, :]
        im_new[single_channel_tif > threshold] = 1 # Nuclei

        output_path = os.path.join(processed_tgt_label_path, im)

        tif.imwrite(output_path, im_new)
