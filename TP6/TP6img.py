import cv2
import numpy as np


def alpha_mask(frame, img_with_alpha, orig=None, threshold=0):
    """Insert an image with alpha channel into another image. """
    # frame: the image to insert into
    # img_with_alpha: the image to be inserted
    # orig: the position to insert the image
    # threshold: the threshold to apply the mask
    # return: the modified image

    # get the size of the frame
    w, h, _ = frame.shape

    # if orig is None, set it to the top left of the frame
    if orig is None:
        orig = [0, 0]
    # if the position is out of the frame, return the frame
    if not (0 <= orig[0] < w and 0 <= orig[1] < h):
        return frame

    # get the start position to insert the image
    x_start, y_start = orig
    # if the position is out of the frame, set it to the top left of the frame
    x_start, y_start = max(x_start, 0), max(y_start, 0)

    # idem for the end position
    x_end, y_end = x_start + img_with_alpha.shape[0], y_start + img_with_alpha.shape[1]
    x_end, y_end = min(x_end, w), min(y_end, h)

    # create the mask
    mask = np.zeros((w, h, 4), dtype=np.uint8)
    # insert the image into the mask
    mask[x_start:x_end, y_start:y_end, :] = img_with_alpha[:x_end - x_start, :y_end - y_start, :]

    # apply the mask to the frame
    np.copyto(frame, mask[:, :, :3], where=(mask[:, :, 3] > threshold)[:, :, None])

    return frame


cap = cv2.VideoCapture(2)
ret, frame = cap.read()
img = cv2.imread('mustache.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, frame = cap.read()
    frame = alpha_mask(frame, img, orig=[100, 100])
    if ret:
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break