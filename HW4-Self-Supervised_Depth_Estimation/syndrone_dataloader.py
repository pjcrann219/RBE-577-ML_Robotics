from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import DataLoader

from monodepth2.datasets.mono_dataset import MonoDataset


class SyndroneDataset(MonoDataset):
    """Superclass for different types of Syndrone dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(SyndroneDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.

        # 1920x1080 images
        height = 1080
        width = 1920

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.K[0, :] /= width
        self.K[1, :] /= height

        self.full_res_shape = (1920, 1080)  #images are 1920 x 1080
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def get_image_path(self, folder, frame_index, side=False): #data path is located in the rgb folder of the depth set we are training
        if frame_index < 0:
            frame_index = 0
        image_path = os.path.join(self.data_path, "{:05d}.jpg".format(frame_index))
        return image_path
    
    def get_color(self, folder, frame_index, side, do_flip):
        # Get the path to the color image based on the frame index
        color = self.loader(self.get_image_path(None, frame_index, None))

        # Optionally flip the image horizontally
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color
    
    def check_depth(self):
        """Check if depth data exists for this dataset"""
        # Get first filename to test
        frame_index = 0
        depth_path = os.path.join(
            self.data_path.replace('rgb', 'semantic'),  # Assuming depth is in parallel folder
            f"{frame_index:05d}.png"                 # Match your depth filename format
        )
        return False
        return os.path.isfile(depth_path)


data_path = 'data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb'
height = 1080
width = 1920
frame_idxs = [-1, 0, 1]
num_scales = 4
is_train= False
img_ext= '.jpg'

# load in 20m rgb
import os
filepath = 'data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb'
filenames = sorted(os.listdir(filepath))

print(filenames[0])

dataset = SyndroneDataset(data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg')

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)

# Test the dataloader
print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")

# Load and inspect first batch
for batch_idx, data in enumerate(dataloader):
    print("\nBatch:", batch_idx)
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            print(f"{key}: shape {data[key].shape}")
    break  # Just show first batch