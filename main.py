import numpy as np
import cv2
import os


def white_circle_remover(image_path):
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.imread(image_path)
    # make a mask to remove circle
    _, mask = cv2.threshold(cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY), 252, 255, cv2.THRESH_BINARY_INV)
    mod = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    erase = mask - mod
    dilation = cv2.dilate(erase, kernel, iterations=1)
    dst = cv2.inpaint(image, dilation, 3, cv2.INPAINT_TELEA)
    return dst



if __name__ == '__main__':
    image_path = r"Samples\original.jpg"
    removed_circle_image = white_circle_remover(image_path)
    cv2.imwrite(r'Samples\output.jpg', removed_circle_image)


