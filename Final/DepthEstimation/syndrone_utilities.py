import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.transforms import Compose
import cv2
import DPT.util.io as io

from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from DPT.dpt.models import DPTDepthModel

def load_model(weights="dpt_large", device=None, eval=False):
    # Function to load model given weights and device
    if weights == "dpt_large":
        weights = "DepthEstimation/DPT/weights/dpt_large-midas-2f21e586.pt"
    elif weights == "dpt_hybrid":
        print("not yet implemented")
    
    # Load model with desired weights
    model = DPTDepthModel(
        path=weights,
        backbone="vitl16_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set eval
    if eval:
        model.eval()

    print(f"Loaded model with {weights}")

    return model

class SyndroneDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, rgb_transform=transforms.ToTensor(), depth_transform=transforms.ToTensor(), split="train"):
        """
        Args:
            rgb_dir (str): Directory with RGB images.
            depth_dir (str): Directory with depth maps.
            transform (callable, optional): Optional transform to be applied on RGB images.
            target_transform (callable, optional): Optional transform to be applied on depth maps.
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.rgb_files = sorted(os.listdir(rgb_dir))
        self.depth_files = sorted(os.listdir(depth_dir))
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        self.split_file = f"data/splits/{split}.txt"
        with open(self.split_file) as f:
            valid_indices = set(int(x.strip()) for x in f.readlines())

        self.rgb_files = [f for i, f in enumerate(self.rgb_files) if i in valid_indices]
        self.depth_files = [f for i, f in enumerate(self.depth_files) if i in valid_indices]

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Load RGB image and depth map
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        
        rgb_image = io.read_image(rgb_path)
        rgb_inputs = self.rgb_transform({"image":rgb_image})["image"]
        
        depth_map = Image.open(depth_path)
        depth_map = np.asarray(depth_map, dtype=np.float32) 
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_map = cv2.resize(np.array(depth_map), (rgb_inputs.shape[2], rgb_inputs.shape[1]), interpolation=cv2.INTER_CUBIC)
        depth_map = np.clip(depth_map, depth_min, depth_max) # Interpolation causes values to be outisde of original range, so clip
        depth_map = 1 / (depth_map)

        return rgb_inputs, depth_map

def plot_tensors(tensor1, tensor2, tensor3):
    tensor1 = tensor1.squeeze(0).permute(1, 2, 0).numpy()  # (3, n, m) -> (n, m, 3)
    tensor2 = tensor2.squeeze(0).numpy()  # (n, m)
    tensor3 = tensor3.squeeze(0).numpy()  # (n, m)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the first tensor (in color)
    axs[0].imshow(tensor1)
    axs[0].axis('off')
    axs[0].set_title('Color Image')

    # Plot the second tensor (grayscale)
    axs[1].imshow(tensor2, cmap='magma')#, vmax=0.5 * tensor2.max())
    axs[1].axis('off')
    axs[1].set_title('Grayscale Image 1')

    # Plot the third tensor (grayscale)
    axs[2].imshow(tensor3, cmap='magma')#, vmax=0.5 * tensor3.max())
    axs[2].axis('off')
    axs[2].set_title('Grayscale Image 2')

    # Show the figure
    plt.tight_layout()
    plt.show()

def SyndroneDataloader(rgb_dir = 'data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb',
                       depth_dir = 'data/Town01_Opt_120_depth/Town01_Opt_120/ClearNoon/height20m/depth',
                       batch_size = 1, shuffle=False, split="train"):

    net_w = net_h = 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    rgb_transform = Compose([
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ])
    
    # depth_transform = ResizeDepth()

    dataset = SyndroneDataset(rgb_dir=rgb_dir, depth_dir=depth_dir, rgb_transform=rgb_transform, split=split)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    
    print(f"Syndrone Dataloader loaded for {split} with batch: {batch_size}, shuffle: {shuffle}, len: {len(dataloader)}")

    return dataloader

def align_pred(pred, m = 0.000025172, b = 0.000033326):
     aligned_prediction = m  * pred + b
     return aligned_prediction

def eigen_loss(outputs, truths, lam=0.5):
    # Input: output and truth values (1/distance)
    outputs_d = 1 / (outputs + 10**-4.5)
    truths_d = 1 / (truths + 10**-4.5)

    # Scale-Invariant MSE + L2
    d = torch.log(outputs_d) - torch.log(truths_d)
    n = d.numel()
    scale_invariant_MSE = torch.sum(d**2) / n - lam*(torch.sum(d)/n)**2

    return scale_invariant_MSE