import matplotlib.pyplot as plt
from util.segmentation.clipseg import CLIPSeg

from PIL import Image
import requests

import numpy as np

import matplotlib.pyplot as plt

thresh = 0.4


prompts = [
    "napkin",
    "plate",
    "pancake",
    "wood"
]

clipseg = CLIPSeg(
    model_path="data/models/clipseg-rd64-refined",
    prompts=prompts
)

url = "https://unsplash.com/photos/8Nc_oQsc2qQ/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjcxMjAwNzI0&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)

preds = clipseg.get_segmentation(image)


# Show plots in 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(image)
    ax.imshow(np.maximum(preds[i], thresh), cmap="viridis", alpha=0.75)
    ax.axis("off")
    ax.set_title(f"{prompts[i]}; {preds[i].shape}")

plt.show()
