import numpy as np
import tifffile as tif
from batchgenerators.utilities.file_and_folder_operations import *


if __name__ == '__main__':

    path = '/Users/17914/phd/Cascade_Tr5_Ts234/'
    t_path = '/Users/17914/phd/Cascade_Tr5_Ts234_255/'
    maybe_mkdir_p(t_path)
    images = subfiles(path, suffix='.tif', sort=True, join=False)
    for i, im in enumerate(images):
        target_name = '255_cascade' + im
        cur_tif = tif.imread(path + im)
        im_0 = np.zeros_like(cur_tif)
        im_1 = np.zeros_like(cur_tif)
        im_2 = np.zeros_like(cur_tif)
        im_0[cur_tif == 1] = 255
        im_0[cur_tif != 1] = 0
        im_1[cur_tif == 2] = 255
        im_1[cur_tif != 2] = 0
        im_2[cur_tif == 3] = 255
        im_2[cur_tif != 3] = 0
        new_im = np.stack([im_0, im_1, im_2], axis=1)
        output_path = os.path.join(t_path, target_name)
        tif.imwrite(output_path, new_im)

