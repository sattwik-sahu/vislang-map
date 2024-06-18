from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from os import path
from typing import Tuple, List
from helpers.activation import sigmoid
import numpy as np
import torch
from torch.functional import F
from PIL import Image


def load_model(model_path: str) -> Tuple[CLIPSegForImageSegmentation, CLIPSegProcessor]:
    """
    Loads the CLIPSeg model and processor into memory.
    Uses local path if available, else downloads from HuggingFace.

    Args:
        model_path (str): Path to the model. Saved here if downloaded from HuggingFace.

    Returns:
        Tuple[CLIPSegForImageSegmentation, CLIPSegProcessor]: Model and processor.
    """
    if path.exists(model_path):
        # Download from HuggingFace
        model = CLIPSegForImageSegmentation.from_pretrained(model_path)
        processor = CLIPSegProcessor.from_pretrained(model_path)
    else:
        # Load from local
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd16")
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd16")

        # Save model and processor
        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)

    return model, processor


class CLIPSeg:
    def __init__(self, model_path: str, prompts: List[str]):
        self.model, self.processor = load_model(model_path)
        self.prompts = prompts
        self.image: Image = None

    def _predict(self, image: Image) -> torch.Tensor:
        """
        Returns the prediction tensor.

        Args:
            image (Image): Image to predict.

        Returns:
            torch.Tensor: Prediction tensor.
        """
        X = self.processor(
            text=self.prompts,
            images=[image] * len(self.prompts),
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            y = self.model.forward(**X)
        return y

    def _reshape_preds(self, y: torch.Tensor, original_image: Image) -> torch.Tensor:
        """
        Reshapes the prediction tensor.

        Args:
            y (torch.Tensor): Prediction tensor.

        Returns:
            torch.Tensor: Reshaped prediction tensor.
        """
        reshaped_pred = F.interpolate(
            y.logits.unsqueeze(1),
            size=(original_image.height, original_image.width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)
        return reshaped_pred

    def get_segmentation(self, image: Image) -> np.ndarray:
        """
        Returns the segmented image.

        Args:
            image (Image): Image to segment.

        Returns:
            Image: Segmented image.
        """
        preds = self._predict(image)
        reshaped_pred = self._reshape_preds(preds, original_image=image)
        probs = reshaped_pred.sigmoid()
        return reshaped_pred.cpu().numpy()
