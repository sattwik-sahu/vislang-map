from matplotlib.animation import FuncAnimation
import torch
import rospy
import cv2
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import pyrealsense2
import message_filters

# import plotly.graph_objects as go
# import numpy as np

# def create_3d_plot(coords):
#     """
#     Create a 3D scatter plot using Plotly.

#     Parameters:
#     - coords: numpy array of shape (n, 3) containing x, y, and z coordinates.
#     """
#     # Extract x, y, z coordinates from the input array
#     x_coords = coords[:, 0]
#     y_coords = coords[:, 1]
#     z_coords = coords[:, 2]

#     fig = go.Figure()

#     # Add 3D scatter plot trace
#     fig.add_trace(go.Scatter3d(
#         x=x_coords,
#         y=y_coords,
#         z=z_coords,
#         mode='markers',
#         marker=dict(
#             size=5,
#             color=z_coords,  # Color by z-coordinate
#             colorscale='Viridis',  # Choose a colorscale
#             opacity=0.8
#         )
#     ))

#     # Update layout
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X Axis',
#             yaxis_title='Y Axis',
#             zaxis_title='Z Axis'
#         ),
#         title='3D Scatter Plot'
#     )

#     # Show the plot
#     fig.show()


def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    _intrinsics = pyrealsense2.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = pyrealsense2.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.D]
    depth = depth[x, y]
    result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
    #result[0]: right, result[1]: down, result[2]: forward
    # This is in robot frame
    return result[2], -result[0], -result[1]
    # return result[2], result[1], -result[0]


def get_pixel_coords(mask: np.ndarray) -> list[list[int]]:
    """
    Get the pixel coordinates of the mask.

    Args:
        mask (np.ndarray): Mask.

    Returns:
        list[list[int]]: List of pixel coordinates.
    """
    coords = np.argwhere(mask)
    return [[x, y] for x, y in coords]


def get_3d_points(pixel_coords: list[list[int]], depth_image: np.ndarray, camera_info) -> np.ndarray:
    """
    Gets the set of 3D points corresponding to the pixel coordinates in the depth image.

    Args:
        pixel_coords (list[list[int]]): List of pixel coordinates.
        depth_image (np.ndarray): Depth image.
        camera_info: Camera information.

    Returns:
        np.ndarray: 3D points.
    """
    points = []
    for coord in pixel_coords:
        x, y = coord
        point = convert_depth_to_phys_coord_using_realsense(x, y, depth_image, camera_info)
        points.append(point)
    return np.array(points)



# Initialize the model
def load_model():
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=16)
    model.eval()
    model.load_state_dict(torch.load('weights/rd16-uni.pth', map_location=torch.device('cpu')), strict=False)
    return model

# Export the model to ONNX
def export_model_to_onnx(model, dummy_input, onnx_file_path):
    torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True, opset_version=11,
                      input_names=['input'], output_names=['output'])
    print(f"Model exported to {onnx_file_path}")

# Quantize the ONNX model
def quantize_model(onnx_file_path, quantized_model_path):
    quantize_dynamic(onnx_file_path, quantized_model_path, weight_type=QuantType.QUInt8)
    print(f"Model quantized and saved to {quantized_model_path}")

# Load the ONNX model
def load_onnx_model(onnx_file_path):
    ort_session = ort.InferenceSession(onnx_file_path)
    return ort_session

# Image transformation pipeline
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])


class cameraInfo:
    def __init__(self, K=None, D=None, height=None, width=None):
        self.K = K 
        self.D = D 
        self.height = height
        self.width = width
    def update(self, K=None, D=None, height=None, width=None):
        self.K = K 
        self.D = D 
        self.height = height
        self.width = width

    

class RosPack:
    def __init__(self, ort_session, transform, prompts):
        self.ort_session = ort_session
        self.transform = transform
        self.prompts = prompts
        self.camera_info = cameraInfo()

        # fig, ax = plt.subplots()
        self.mask = np.zeros((480, 640))
        # self.img = ax.imshow(self.mask, animated=True)
        # plt.title('Segmentation Mask')
        # plt.colorbar(self.img)

        # Initialize CvBridge
        self.bridge = CvBridge()
    
        # Subscribe to the image topic
        # rospy.Subscriber("/camera/color/image_raw", RosImage, self.image_callback,  queue_size=1, buff_size=2**24)
        # rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", RosImage, self.depth_callback, queue_size=1, buff_size=2**24)
        # rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        image_sub = message_filters.Subscriber("/camera/color/image_raw", RosImage)
        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", RosImage)
        camera_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub, camera_info_sub], 10, 0.1)
        ts.registerCallback(self.callback)

        # ani = FuncAnimation(fig, self.update_mask, frames=None, interval=0.1)  # Interval in milliseconds
        # plt.show()


    def update_mask(self, *args):
        self.img.set_array(self.mask)
        plt.draw()
        # plt.pause(0.001)

    def callback(self, image_msg, depth_msg, camera_info_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        print(f"Image shape: {cv_image.shape}")
        input_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        preds = self.process_image(input_image).squeeze(0).squeeze(0).numpy()
        # Resize the image to match the input size of the original image
        preds = cv2.resize(preds, (640, 480), interpolation=cv2.INTER_NEAREST)
        print(f"Prediction shape: {preds.shape}")

        thresh = 0.5
        self.mask = (preds > thresh).astype(np.uint8)


        self.camera_info.update(camera_info_msg.K, camera_info_msg.D, camera_info_msg.height, camera_info_msg.width)

        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        # r = convert_depth_to_phys_coord_using_realsense(300, 300, depth, self.camera_info)
        # print(r)

        # ======== 3D Visualization ========
        pixels = get_pixel_coords(self.mask)
        points = get_3d_points(pixels, depth, self.camera_info)

        print(f"Dimension of points: {points.shape}")

        if points.shape[0] > 7000:
            create_3d_plot(points)
            quit(1)

        # self.visualizer.update_data(points)
        # self.visualizer.update_plot()
        # =================================




    def image_callback(self, ros_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            input_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            preds = self.process_image(input_image)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def process_image(self, input_image):
        img = self.transform(input_image).unsqueeze(0).numpy()
        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        preds = torch.tensor(ort_outs[0])
        return preds

    def depth_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def camera_info_callback(data):
        self.camera_info = data
    

def main():
    rospy.init_node('image_processor', anonymous=True)
    
    
    
    # Prompts
    prompts = ['straight line', 'something to fill', 'wood', 'a jar']
    
    # Load model and transformation pipeline
    # model = load_model()
    transform = get_transform()
    
    # Dummy input for exporting the model
    # dummy_input = torch.randn(1, 3, 352, 352)
    
    # Export model to ONNX
    # onnx_file_path = 'model.onnx'
    # export_model_to_onnx(model, dummy_input, onnx_file_path)
    # Quantize the ONNX model
    quantized_model_path = 'models/clipseg_rd16_quantized.onnx'
    # quantize_model(onnx_file_path, quantized_model_path)

    # Load quantized ONNX model for inference
    ort_session = load_onnx_model(quantized_model_path)
    
    # Load ONNX model for inference
    # ort_session = load_onnx_model(onnx_file_path)
    
    ros_pack = RosPack(ort_session, transform, prompts)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
