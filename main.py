import math
from cv2 import cv2
from matplotlib import pyplot as plt
import pytesseract as ocr
import numpy as np

DIST_THRESHOLD = 9

ocr.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

text = []
component_list = []


def init():
    image = cv2.imread('Example.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_show(gray_image, "gray image")
    return gray_image


def binarize(im):
    _, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_show(thresh, "Binarized image with Otsu's algorithm")
    return thresh


def image_show(image, title):
    plt.title(title)
    plt.imshow(image, cmap='gray', vmin='0', vmax='255')
    plt.show()


def connected_comp_labelling(image):
    # function used to detect the different connected components in our image
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 4)

    barycenter_list = []

    # for loop starts from 1 since the first blob is always the background
    for i in range(1, retval):
        # each integer of the component is given the value 255, hence labeled as foreground
        # component = np.array([[0 if pixel == i else 255 for pixel in row] for row in labels], dtype=np.uint8)
        # image_show(component, f"component {i}")
        barycenter_list.append(centroids[i])

    B = np.zeros((2, 7))
    B_flat = B.flatten()

    for i in range(1, 13):
        dist_centroids = math.sqrt((barycenter_list[i][0] - barycenter_list[i - 1][0]) ** 2 + \
                                   (barycenter_list[i][1] - barycenter_list[i - 1][1]) ** 2)
        if dist_centroids < DIST_THRESHOLD:
            # B_flat[i] = 1
            B_flat[i-1] = 1

    # B = B_flat.reshape((2, 7))
    return B_flat


def group_char(t, b):
    t_new = t.replace('\n', '')
    num_list = [ele for ele in t_new]
    num_list_copy = ['']*len(num_list)
    # print(num_list)
    # print(b)

    for i in range(0, len(num_list)-1):
        if b[i] == 1:
            num_list_copy[i] = num_list[i + 1]
            num_list[i + 1] = ''

    new_num_list = [i + j for i, j in zip(num_list, num_list_copy)]

    new_num_list = [int(i) for i in new_num_list if i != '']

    result = []

    for i in range(0, 5):
        result.append([new_num_list[i], new_num_list[i+5]])

    print(result)


def sharpen(im):
    kernel = np.array([[0, -1, 0],
                       [-1, 6, -1],
                       [0, -1, 0]])
    im = cv2.bilateralFilter(im, 9, 50, 50)
    sharped_im = cv2.filter2D(im, -1, kernel)
    return sharped_im


if __name__ == "__main__":
    gray_im = init()
    gray_im = sharpen(gray_im)
    image_show(gray_im, "filtered")
    binary_im = binarize(gray_im)
    text_ex = ocr.image_to_string(binary_im)
    # print(text_ex)
    B = connected_comp_labelling(binary_im)
    group_char(text_ex, B)
