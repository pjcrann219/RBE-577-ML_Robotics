import os
import cv2
from matplotlib import pyplot as plt
from torchvision.transforms import Compose

from syndrone_utilities import *
import DPT.util.io as io
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

def load_rgb(rgb_path):
    # Load input images
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

    rgb_image = io.read_image(rgb_path)
    rgb_inputs = rgb_transform({"image":rgb_image})["image"]

    return rgb_inputs

def load_depth(depth_path):
    # Load depth images
    depth_map = Image.open(depth_path)
    depth_map = np.asarray(depth_map, dtype=np.float32) 
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map = cv2.resize(np.array(depth_map), (672, 384), interpolation=cv2.INTER_CUBIC)
    depth_map = np.clip(depth_map, depth_min, depth_max) # Interpolation causes values to be outisde of original range, so clip
    depth_map = 1 / (depth_map)

    return depth_map

def save_frame(path, input, truth, pred_orig, pred, num):
    # Save each frame as img

    os.makedirs(path, exist_ok=True)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(np.transpose(input, (1, 2, 0)))
    axs[0].axis('off')
    axs[0].set_title('Input Image')

    axs[1].imshow(np.array(pred_orig).squeeze(), cmap='magma')
    axs[1].axis('off')
    axs[1].set_title('Pre-trained Predicted Depth Map')

    axs[2].imshow(np.array(pred).squeeze(), cmap='magma')
    axs[2].axis('off')
    axs[2].set_title('Fine-tuned Predicted Depth Map')

    axs[3].imshow(np.array(truth).squeeze(), cmap='magma')
    axs[3].axis('off')
    axs[3].set_title('Truth Depth Map')


    plt.tight_layout()
    filename = os.path.join(path, f'img_{num}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def save_as_video(image_folder, output_video, fps=10):
    # Save frame images as a mp4 video
    def get_number(filename):
        return int(filename.split('_')[1].split('.')[0])

    images = sorted(
        [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".png")],
        key=lambda x: get_number(os.path.basename(x))
    )

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for img in images:
        frame = cv2.imread(img)
        out.write(frame)

    out.release()
    print(f"Video saved as {output_video}")

def run_testing(model_weights, orig_model, device):
    # Load model with new weights, inference each training frame, save output
    if orig_model == 'dpt_large':
        backbone = "vitl16_384"
    elif orig_model == 'dpt_hybrid':
        backbone = "vitb_rn50_384"
    model = load_model(weights=model_weights,
                    backbone=backbone,
                    device=device,
                    eval=True)
    model_orig = load_model(weights=orig_model,
                    device=device,
                    eval=True)

    dataloader_test = SyndroneDataloader(batch_size=1,shuffle=False, split='test')

    for batch_idx, (input, truth) in enumerate(dataloader_test):
        print(batch_idx)
        with torch.no_grad():
            pred = model(input)
            pred_orig = model_orig(input)
        input, truth, pred, pred_orig = np.array(input).squeeze(), np.array(truth).squeeze(), np.array(pred).squeeze(), np.array(pred_orig).squeeze()

        save_frame(f'DepthEstimation/imgs{model_weights}', input, truth, pred_orig, pred, batch_idx)


# Example usage

# model_weights = 'DepthEstimation/models/2024-11-25 13:55:32/syndrone_weights_16.pt'
# orig_model = 'dpt_large'
 
model_weights = 'DepthEstimation/models/2024-11-26 08:43:27/syndrone_weights_18.pt'
orig_model = 'dpt_hybrid'

# select device
device = "cpu"
print("device: %s" % device)

run_testing(model_weights, orig_model, device)
save_as_video(f'DepthEstimation/imgs{model_weights}', "output_video2.mp4")
