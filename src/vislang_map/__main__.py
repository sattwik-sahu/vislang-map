from src.util.ros.camera.clipseg import CameraCLIPSeg


def main():
    camera_topic = "/nerian_right/image_color"
    camera_node = CameraCLIPSeg(
        name="my_camera_node",
        camera_topic=camera_topic,
        model_path="clipseg-rd64-refined",
        prompts=["grass", "barrier", "sky", "navigable road"],
    )

    camera_node.run()


if __name__ == "__main__":
    main()
