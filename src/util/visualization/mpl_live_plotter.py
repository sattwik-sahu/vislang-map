import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image

class LivePlotter:
    def __init__(self):
        self.fig, (self.image_ax, self.mask_ax) = plt.subplots(1, 2, figsize=(10, 5))
        
        self.image_ax.set_title('Input Image')
        self.mask_ax.set_title('Segmentation Mask')

        self.image_display = self.image_ax.imshow(np.zeros((512, 512, 3)), aspect='auto')
        self.mask_display = self.mask_ax.imshow(np.zeros((512, 512)), cmap='gray', aspect='auto')

        self.image = None
        self.mask = None

        ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()

    def set_data(self, image, mask):
        self.image = image
        self.mask = mask

    def update_plot(self, frame):
        if self.image is not None:
            self.image_display.set_data(self.image)
        if self.mask is not None:
            self.mask_display.set_data(self.mask)
        self.fig.canvas.draw()
