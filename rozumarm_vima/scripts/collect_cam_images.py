import numpy as np
import cv2
import os
import os.path as osp


IMAGE_RESOLUTION = (1280, 1024)


def create_folders(out_folder):
    os.makedirs(osp.join(out_folder, "top"), exist_ok=True)
    os.makedirs(osp.join(out_folder, "front"), exist_ok=True)


def stream_camera(top_cam, front_cam):
    while True:
        top_frame = top_cam()
        front_frame = front_cam()

        # cv2.imshow("top", top_frame)
        # cv2.imshow("front", front_frame)

        # ret = cv2.waitKey(1)
        # if ret == 'q':
        #     exit(0)
        # elif ret != -1:
        return top_frame, front_frame


def collect_dataset(top_cam, front_cam, out_folder):
    # cv2.namedWindow("top", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("top", *IMAGE_RESOLUTION[::-1])
    # cv2.namedWindow("front", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("front", *IMAGE_RESOLUTION)

    image_id = 0
    while True:
        input("Take a shot...")
        top_image, front_image = stream_camera(top_cam, front_cam)

        cv2.imwrite(osp.join(out_folder, f"top/{image_id:04}.png"), top_image)
        cv2.imwrite(osp.join(out_folder, f"front/{image_id:04}.png"), front_image)

        print(f"Pair of images with id {image_id} was written.")

        image_id += 1


if __name__ == "__main__":
    DATASET_DIR = "cam-images"
    TOP_CAM_ID = 0
    FRONT_CAM_ID = 2

    create_folders(DATASET_DIR)

    from rozumarm_vima_utils.camera import Camera
    top_cam = Camera(TOP_CAM_ID, IMAGE_RESOLUTION)
    front_cam = Camera(FRONT_CAM_ID, IMAGE_RESOLUTION)

    def read_top_cam():
        status, img = top_cam.update()
        assert status
        return img
    
    def read_front_cam():
        status, img = front_cam.update()
        assert status
        return img
    
    collect_dataset(read_top_cam, read_front_cam, DATASET_DIR)
