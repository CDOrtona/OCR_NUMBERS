from cv2 import cv2
from matplotlib import pyplot as plt
import pytesseract as ocr

ocr.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def init():
    image = cv2.imread('Example.png')
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    text = ocr.image_to_string(gray_image)
    print((text))
    image_show(gray_image, "gray image")


def image_show(image, title):
    plt.title(title)
    plt.imshow(image, cmap='gray', vmin='0', vmax='255')
    plt.show()


if __name__ == "__main__":
    init()


