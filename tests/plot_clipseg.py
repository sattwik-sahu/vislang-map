import matplotlib.pyplot as plt

from src.util.segmentation import CLIPSeg

thresh = 0.5

# Show plots in 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
  ax.imshow(image)
  ax.imshow(preds[i] > thresh, cmap="viridis", alpha=0.75)
  ax.axis("off")
  ax.set_title(f"{prompts[i]}; {preds[i].shape}")
