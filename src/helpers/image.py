import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sensor_msgs.msg import Image as ROSImage
from PIL.Image import Image as PILImage, fromarray as numpy_to_pil
import cv2
from cv_bridge import CvBridge


class ImageConverter:
    @staticmethod
    def from_pil_to_numpy(image: PILImage) -> np.ndarray:
        """
        Converts a PIL Image to a numpy array.

        Args:
            image (PILImage): Image to convert.

        Returns:
            np.ndarray: Numpy array.
        """
        return np.array(image)

    @staticmethod
    def from_numpy_to_pil(image: np.ndarray) -> PILImage:
        """
        Converts a numpy array to a PIL Image.

        Args:
            image (np.ndarray): Numpy array to convert.

        Returns:
            PILImage: PIL Image.
        """
        return numpy_to_pil(image)

    @staticmethod
    def from_pil_to_tensor(image: PILImage) -> torch.Tensor:
        """
        Converts a PIL Image to a PyTorch tensor.

        Args:
            image (PILImage): Image to convert.

        Returns:
            torch.Tensor: PyTorch tensor.
        """
        return F.to_tensor(image)

    @staticmethod
    def from_tensor_to_pil(image: torch.Tensor) -> PILImage:
        """
        Converts a PyTorch tensor to a PIL Image.

        Args:
            image (torch.Tensor): PyTorch tensor to convert.

        Returns:
            Image: PILImage.
        """
        return F.to_pil_image(image)

    @staticmethod
    def from_ros_to_numpy(image: ROSImage) -> np.ndarray:
        """
        Converts a ROS Image to a numpy array.

        Args:
            image (ROSImage): ROS Image to convert.

        Returns:
            np.ndarray: Numpy array.
        """
        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")

    @staticmethod
    def from_numpy_to_ros(image: np.ndarray, encoding: str = "bgr8") -> ROSImage:
        """
        Converts a numpy array to a ROS Image.

        Args:
            image (np.ndarray): Numpy array to convert.
            encoding (str): Encoding of the ROS Image. Default is 'bgr8'.

        Returns:
            ROSImage: ROS Image.
        """
        bridge = CvBridge()
        return bridge.cv2_to_imgmsg(image, encoding=encoding)

    @staticmethod
    def from_ros_to_pil(image: ROSImage) -> PILImage:
        """
        Converts a ROS Image to a PIL Image.

        Args:
            image (ROSImage): ROS Image to convert.

        Returns:
            Image: PIL Image.
        """
        # Convert ROS Image message to OpenCV image
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

        # Convert OpenCV image to PIL Image
        return numpy_to_pil(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def from_pil_to_ros(image: PILImage, encoding: str = "bgr8") -> ROSImage:
        """
        Converts a PIL Image to a ROS Image.

        Args:
            image (PILImage): PIL Image to convert.
            encoding (str): Encoding of the ROS Image. Default is 'bgr8'.

        Returns:
            ROSImage: ROS Image.
        """
        # Convert the PIL Image to a numpy array
        image_np = np.array(image)

        # Convert the numpy array to a ROS Image
        ros_image = bridge.cv2_to_imgmsg(image_np, encoding='bgr8')
