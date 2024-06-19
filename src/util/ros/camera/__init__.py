import rospy
from sensor_msgs.msg import Image as ROSImage
from PIL.Image import Image as PILImage
from cv_bridge import CvBridge
import cv2
from abc import ABC, abstractmethod
from src.helpers.image import ImageConverter
from functools import partialmethod


class CameraProcessorNode(ABC):
    """
    Class to process images from a camera feed.
    Runs as a ROS node.
    """

    def __init__(self, name: str, camera_topic: str):
        rospy.init_node(name)

        # Subscribe to the first image topic
        self.image_sub = rospy.Subscriber(
            camera_topic, ROSImage, lambda image: self.image_callback(image)
        )

    def image_callback(self, image: ROSImage):
        print("Hello")
        try:
            # Convert ROS Image message to OpenCV image
            pil_image = ImageConverter.from_ros_to_pil(image=image)
            self.callback(pil_image)
        except Exception as e:
            rospy.logerr("Error converting image: %s" % str(e))

    @abstractmethod
    def callback(self, image: PILImage) -> None:
        """
        Callback function to process the image.

        Args:
            image (PILImage): Image to process.
        """
        pass

    def run(self):
        rospy.spin()
