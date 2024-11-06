from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


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
        # h = 1080
        # w = 1920
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
    def get_image_path(self, folder, frame_index, side): #data path is located in the rgb folder of the depth set we are training
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, "rgb", "{:05d}.jpg".format(frame_index))
        return image_path
    def get_color(self, frame_index, do_flip):
        # Get the path to the color image based on the frame index
        color = self.loader(self.get_image_path(None, frame_index, None))

        # Optionally flip the image horizontally
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        town_filename = os.path.join(
            self.data_path,
            scene_name,
            "rgb",
            "{:06d}.jpg".format(int(frame_index)))

        return os.path.isfile(town_filename)



