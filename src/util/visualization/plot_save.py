import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from datetime import datetime


def plot_segmentation_masks(
    image: Image, masks: np.ndarray, titles: list, output_folder: str
):
    assert len(titles) == masks.shape[0], "Number of titles must match number of masks"

    h, w = image.size
    n = masks.shape[0]

    # Create a 1xn+1 grid of subplots
    fig, axs = plt.subplots(1, n + 1, figsize=(15, 5))

    # Plot the input image
    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    # Plot each mask overlayed on the image
    for i in range(n):
        axs[i + 1].imshow(image)
        axs[i + 1].imshow(
            masks[i], cmap="jet", alpha=0.5
        )  # Overlay the mask with transparency
        axs[i + 1].set_title(titles[i])
        axs[i + 1].axis("off")

    # Adjust layout and save the figure
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"segmentation_{timestamp}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Figure saved to {output_path}")
