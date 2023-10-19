from rozumarm_vima_utils.camera import Camera, CamDenseReader
import numpy as np
import cv2

def toTuple(a, b):
    return [((*ai,), (*bi,)) for ai, bi in zip(a, b) ]

from detect_boxes import detect_boxes_by_aruco, detect_boxes_by_segmentation, detect_boxes_visual
from calibrate_table import calibrate_table_by_aruco, calibrate_table_by_markers
from params import table_aruco_size, box_aruco_size, box_size, K, D, target_table_markers


class CubeDetector():
    def __init__(self) -> None:
        self.cam = Camera(0, (1280, 1024))
        ret, image = self.cam.update()
        self.table_frame, _ = calibrate_table_by_aruco(image, "top", K, D, table_aruco_size)

    
    def detect(self):
        ret, image = self.cam.update()
        print(image.shape)
        cv2.imwrite('./test.jpg', image)
        boxes_positions, boxes_orientations = detect_boxes(image, "top", K, D, self.table_frame, box_aruco_size, box_size)
        print(f"Detected {len(boxes_positions)} boxes")
        res = toTuple( boxes_positions, boxes_orientations)
        for i, j in enumerate(res):
            print(f'cube #{i}: {j}')
        return res


import time
class CubeDenseDetector():
    def __init__(self) -> None:
        self.cam_1 = CamDenseReader(2, 'cam_top_video.mp4')
        self.cam_2 = CamDenseReader(4, 'cam_front_video.mp4')
        self.cam_1.start_recording()
        self.cam_2.start_recording()

        self.view = "top"
        self.working_cam = self.cam_1

        ret, image = self.working_cam.read_image()
        # self.table_frame, _ = calibrate_table_by_aruco(image, self.view, K, D, table_aruco_size)
        self.table_transform = calibrate_table_by_markers(image, self.view, K, D, target_table_markers)
        time.sleep(3)

    def detect(self):
        ret, image = self.working_cam.read_image()
        print(image.shape)
        cv2.imwrite('./test.jpg', image)
        # boxes_positions, boxes_orientations = detect_boxes_by_aruco(image, self.view, K, D, self.table_frame, box_aruco_size, box_size)
        # boxes_positions, boxes_orientations = detect_boxes_by_segmentation(image, self.view, K, D, self.table_frame, box_size)
        boxes_positions, boxes_orientations = detect_boxes_visual(image, self.view, K, D, self.table_transform)
        print(f"Detected {len(boxes_positions)} boxes")
        res = toTuple( boxes_positions, boxes_orientations)
        for i, j in enumerate(res):
            print(f'cube #{i}: {j}')
        return res

    def release(self):
        self.cam_1.stop_recording()
        self.cam_2.stop_recording()


# detector = CubeDetector()
detector = CubeDenseDetector()
