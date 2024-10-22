import os

import tifffile as tif
from PIL import Image
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles

if __name__ == '__main__':

    path = '/data/zucksliu/Glom-Segmnentation/media/fabian/label_gt_FullStack/' #Edit input path
    t_path = '/data/zucksliu/Glom-Segmnentation/media/fabian/labelTr/'  #Edit target path
    threshold = 250
    images = subfiles(path, suffix='.tif', sort=True, join=False)

    ## Please Edit following values for specific channels.
    for i, im in enumerate(images):
        cur_tif = tif.imread(path + im)
        num_channels = cur_tif.shape[1]
        im_new = np.zeros_like(cur_tif[:, 0, :, :])
        single_channel_tif = cur_tif[:, 3, :, :]
        im_new[single_channel_tif > threshold] = 5 #Blood
        single_channel_tif = cur_tif[:, 0, :, :]
        im_new[single_channel_tif > threshold] = 4 # Brown Space
        single_channel_tif = cur_tif[:, 2, :, :]
        im_new[single_channel_tif > threshold] = 3 # Capsule
        single_channel_tif = cur_tif[:, 1, :, :]
        im_new[single_channel_tif > threshold] = 2 # GBM
        single_channel_tif = cur_tif[:, 4, :, :]
        im_new[single_channel_tif > threshold] = 1 # Nuclei
        output_path = os.path.join(t_path, im)
        tif.imwrite(output_path, im_new)
