from src.util.ros.camera import CameraProcessorNode
from src.util.segmentation.clipseg import CLIPSeg
from src.util.visualization.plot_save import plot_segmentation_masks
from PIL.Image import Image as PILImage
from typing import List


class CameraCLIPSeg(CameraProcessorNode):
    """
    ROS node to segment images from ROS topic using CLIPSeg.
    """

    def __init__(self, name: str, camera_topic: str, model_path: str, prompts: List[str]):
        super().__init__(name=name, camera_topic=camera_topic)
        self.prompts = prompts
        self.clipseg = CLIPSeg(model_path=model_path, prompts=prompts)

    def callback(self, image: PILImage) -> None:
        seg_probs = self.clipseg.get_segmentation(image=image)
        plot_segmentation_masks(image=image, masks=seg_probs, titles=self.prompts, output_folder='data/out')
        print(seg_probs.shape, seg_probs.min(), seg_probs.max())
