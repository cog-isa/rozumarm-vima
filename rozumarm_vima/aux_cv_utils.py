import numpy as np
import cv2


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def crop_front_image(image):
    image = image[200:780, 50:1220]
    image = rotate_image(image, 0.5)
    #image = cv2.resize(image, (256, 128))
    return image


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def preprocess_top_image(image):
    """
    image from cam -> obs['rgb]['top']
    """
    image = rotate_image(image, 1.5)
    image = image[170:800, 70:1210]
    image = image[::-1]
    image = cv2.resize(image, (256, 128))
    return image
