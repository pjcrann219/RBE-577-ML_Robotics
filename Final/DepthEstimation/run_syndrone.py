import os
import glob
import torch
import cv2
import argparse
from matplotlib import pyplot as plt

import DPT.util.io as io

from torchvision.transforms import Compose

from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

from syndrone_utilities import *

optimize = False

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("device: %s" % device)

# load network
model_type = "dpt_large" # DPT-Large
model_path = "DepthEstimation/DPT/weights/dpt_large-midas-2f21e586.pt"

net_w = net_h = 384
model = DPTDepthModel(
    path=model_path,
    backbone="vitl16_384",
    non_negative=True,
    enable_attention_hooks=False,
)

normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

transform = Compose(
    [
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
    ]
)

if optimize == True and device == torch.device("cuda"):
    model = model.to(memory_format=torch.channels_last)
    model = model.half()

model.to(device)
model.eval()

print(f"{model_type} loaded from {model_path}")

input_path = "data/Town01_Opt_120_color/Town01_Opt_120/ClearNoon/height20m/rgb"
img_names = sorted(glob.glob(os.path.join(input_path, "*")))
num_images = len(img_names)

output_path = "DepthEstimation/output"
os.makedirs(output_path, exist_ok=True)

print("start processing")

for ind, img_name in enumerate(img_names[:50]):
    if os.path.isdir(img_name):
        continue

    print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

    img = io.read_image(img_name)
    img_input = transform({"image": img})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        # img_input.shape
        # (3, 384, 672)
        # sample.shape
        # torch.Size([1, 3, 384, 672])
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction0 = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction0.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        )
    print(f"\t{img.shape}, {img_input.shape}, {prediction0.shape}, {prediction.shape}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(np.transpose(img_input, (1, 2, 0)))
    axs[0].axis('off')
    axs[0].set_title('Input Image')

    # Plot the second tensor (grayscale)
    axs[1].imshow(prediction0.cpu().squeeze(), cmap='magma')#, vmax=0.5 * tensor2.max())
    axs[1].axis('off')
    axs[1].set_title('Raw Prediction')

    axs[2].imshow(prediction.squeeze(), cmap='magma')#, vmax=0.5 * tensor2.max())
    axs[2].axis('off')
    axs[2].set_title('Interpolated Prediction')

    print(f"\tRaw Prediction: min({prediction0.min():.2f}) max({prediction0.max():.2f}) avg({prediction0.mean():.2f})")
    print(f"\Int Prediction: min({prediction.min():.2f}) max({prediction.max():.2f}) avg({prediction.mean():.2f})")
    plt.show()

    filename = os.path.join(
        output_path, os.path.splitext(os.path.basename(img_name))[0]
        )   
    io.write_depth(filename, prediction, bits=2, absolute_depth=False)

print("finished")