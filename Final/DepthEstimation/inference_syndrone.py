import os
import cv2
from matplotlib import pyplot as plt
from torchvision.transforms import Compose

from syndrone_utilities import *
import DPT.util.io as io
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

def show_frame(input, truth, pred):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(np.transpose(input, (1, 2, 0)))
    axs[0].axis('off')
    axs[0].set_title('Input Image')

    axs[1].imshow(np.array(pred).squeeze(), cmap='magma')
    axs[1].axis('off')
    axs[1].set_title('Fine-tuned Predicted Depth Map')

    axs[2].imshow(np.array(truth).squeeze(), cmap='magma')
    axs[2].axis('off')
    axs[2].set_title('Truth Depth Map')


    plt.tight_layout()
    plt.show()

def run_inference(model_weights):
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # Load model
    model = load_model(weights=model_weights,
                    device=device,
                    eval=True)

    # Load test dataloder
    dataloader_test = SyndroneDataloader(batch_size=1,shuffle=True, split='test')

    # Loop and inference
    for batch_idx, (input, truth) in enumerate(dataloader_test):
        print(batch_idx)
        input = input.to(device)
        with torch.no_grad():
            pred = model(input)
        input = np.array(input.cpu()).squeeze()
        truth = np.array(truth).squeeze()
        pred = np.array(pred.cpu()).squeeze()

        show_frame(input, truth, pred)

if __name__ == "__main__":
    model_weights = 'DepthEstimation/fine_tuned_models/dpt_large_fine_tuned/syndrone_weights_16.pt'
    run_inference(model_weights=model_weights)