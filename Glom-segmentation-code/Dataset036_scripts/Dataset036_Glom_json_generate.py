from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import tifffile as tif
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


"""
   Generates a dataset.json file in the output folder

   channel_names:
       Channel names must map the index to the name of the channel, example:
       {
           0: 'T1',
           1: 'CT'
       }
       Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

   labels:
       This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
       Example regular labels:
       {
           'background': 0,
           'left atrium': 1,
           'some other label': 2
       }
       Example region-based training:
       {
           'background': 0,
           'whole tumor': (1, 2, 3),
           'tumor core': (2, 3),
           'enhancing tumor': 3
       }

       Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

   num_training_cases: is used to double check all cases are there!

   file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
   segmentations!

   dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
   completeness and as a reminder that these would be great!

   overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
   BaseReaderWriter, place it into nnunet.imageio and reference it here by name

   kwargs: whatever you put here will be placed in the dataset.json as well

   """

if __name__ == '__main__':

    dataset_name = 'Dataset040_GlomHovTest'  # Needed to be changed with specific output dataset # Output dir path
    cur_path = '/data/zucksliu/Glom-Segmnentation/media/fabian' # Input dir path

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    # we extract the downloaded train and test datasets to two separate folders and name them Fluo-C3DH-A549-SIM_train
    # and Fluo-C3DH-A549-SIM_test
    train_source = cur_path + '/train/'
    test_source = cur_path + '/test/'
    lbTr_path = cur_path + '/labelTr/'
    lbTs_path = cur_path + '/labelTs/'


    # with the old nnU-Net we had to convert all the files to nifti. This is no longer required. We can just copy the
    # tif files

    # tif is broken when it comes to spacing. No standards. Grr. So when we use tif nnU-Net expects a separate file
    # that specifies the spacing. This file needs to exist for EVERY training/test case to allow for different spacings
    # between files. Important! The spacing must align with the axes.
    # Here when we do print(tifffile.imread('IMAGE').shape) we get (29, 300, 350). The low resolution axis is the first.
    # The spacing on the website is given in the wrong axis order. Great.

    '''
    Spacaing is based on the scale of image. Could find it in ImageJ Fiji > Image > Properties with X, Y, Z spacing. 
    '''
    spacing = (1, 0.2136657, 0.2136657)  # Could be found in the ImageJ tool for specific X Y Z spacing value

    # train set
    # if we were to be super clean we would go by IDs but here we just trust the files are sorted the correct way.
    # Simpler filenames in the cell tracking challenge would be soooo nice.

    # split train raw data channels into multiple image and stored into specific train dir
    images = subfiles(train_source, suffix='.tif', sort=True, join=False)
    for i, im in enumerate(images):
        target_name = f'Glom_{i:03d}' # Glom ID start with 0, 1, 2...
        cur_tif = tif.imread(train_source + im)
        num_channels = cur_tif.shape[1]
        save_json({'spacing': spacing}, join(imagestr, target_name + '.json'))
        for j in range(num_channels):
            single_channel_tif = cur_tif[:, j, :, :]
            output_path = os.path.join(imagestr, target_name + f'_{j:04d}' + '.tif') # Glom_[ID num]_[Ch num]
            tif.imwrite(output_path, single_channel_tif)


    # split test raw data channels into multiple images and stored into specific test dir
    images = subfiles(test_source, suffix='.tif', sort=True, join=False)
    for i2, im in enumerate(images):
        target_name = f'Glom_{i+i2+1:03d}' # Test Glom ID start with [last Train ID +1], [last Train ID +2] ...
        cur_tif = tif.imread(test_source + im)
        num_channels = cur_tif.shape[1]
        save_json({'spacing': spacing}, join(imagests, target_name + '.json'))
        for j in range(num_channels):
            single_channel_tif = cur_tif[:, j, :, :]
            output_path = os.path.join(imagests, target_name + f'_{j:04d}' + '.tif')
            tif.imwrite(output_path, single_channel_tif)
    
    # move trained labels into trained label dir
    images = subfiles(lbTr_path, suffix='.tif', sort=True, join=False)
    for i, im in enumerate(images):
        target_name = f'Glom_{i:03d}' # Corresponding Train Glom ID labels
        save_json({'spacing': spacing}, join(labelstr, target_name + '.json'))
        cur_tif = tif.imread(lbTr_path + im)
        output_path = os.path.join(labelstr, target_name + '.tif')
        tif.imwrite(output_path, cur_tif)

    # move test labels into test label dir
    images = subfiles(lbTs_path, suffix='.tif', sort=True, join=False)
    for i2, im in enumerate(images):
        target_name = f'Glom_{i+i2+1:03d}' # Corresponding Test Glom ID labels
        save_json({'spacing': spacing}, join(labelsts, target_name + '.json'))
        cur_tif = tif.imread(lbTs_path + im)
        output_path = os.path.join(labelsts, target_name + '.tif')
        tif.imwrite(output_path, cur_tif)

    # now we generate the dataset json
    print('Generating data json ...')
    generate_dataset_json(
        join(nnUNet_raw, dataset_name),
        {0: 'Nuclei', 1: 'Amine/Proteins', 2: 'carbon-hydrates'}, # Raw Data channels
        {"background": 0, "Nuc": 1, "GBM": 2, "Cap": 3, "Blood Space": 4, "Bowman's Space": 5}, # Ground truth labels
        1,   # would change with training sample nums.
        '.tif',
        None,
        'Glom',
        None,
        '4/3/2023',
        'Kenan Ye',
        'Glom segmentation',
        'Tiff3DIO',
    )
    print('Complete!')
