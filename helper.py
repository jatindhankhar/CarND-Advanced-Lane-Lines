import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle


def undistort(img, mtx, dist):
    """ Undistorts an image """
    return cv2.undistort(img, mtx, dist, None, mtx)


def normalized(img):
    return np.uint8(255 * img / np.max(np.absolute(img)))


def to_RGB(img):
    if img.ndim == 2:
        img_normalized = normalized(img)
        return np.dstack((img_normalized, img_normalized, img_normalized))
    elif img.ndim == 3:
        return img
    else:
        return None


def load_data():
    """Loads the pickle data and returns a tuple"""
    print("Loading the data ...")
    file = open("wide_dist_pickle.p", 'rb')
    object_file = pickle.load(file)
    file.close()
    return (object_file['mtx'], object_file['dist'])


def warp_transform(image, reverse=False):
    """
    Performs perspective transform and warps the image
    Returns warped and transformed image
    If reverse=True it performs reverse transformation
    """

    img_size = (image.shape[1], image.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    if reverse:
        return cv2.warpPerspective(image, M_inv, img_size, flags=cv2.INTER_LINEAR)
    else:
        return cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)

    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def hls_select(image, thresh=(0, 255)):
    # detect lane lines by saturation
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def combined_threshold(image, sobel_kernel=3):
    # apply multiple threshold functions
    mag_binary = mag_thresh(image, sobel_kernel=sobel_kernel, thresh=(50, 200))
    dir_binary = dir_threshold(
        image, sobel_kernel=sobel_kernel, thresh=(0.7, 1.3))
    s_binary = hls_select(image, thresh=(170, 255))
    combined_binary = np.zeros_like(dir_binary)
    combined_binary[((mag_binary == 1) & (dir_binary == 1))
                    | (s_binary == 1)] = 1
    return combined_binary
