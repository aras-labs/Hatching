import cv2
import numpy as np


def sculpt(image):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    # Read the input image
    # img = cv2.imread('img_bg.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_c = image.copy()

    # Detect faces
    faces = face_cascade.detectMultiScale(image, 1.1, 4)

    # Draw rectangle around the faces
    (x, y, w, h) = faces[0]
    (x_p, y_p, x_pp, y_pp) = (max(x-w//2, 0), max(y-h//2, 0), min(x+w+w//2, image.shape[1]), min(y+h+h//2, image.shape[0]))
    # cv2.rectangle(img_c, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # cv2.rectangle(img_c, (x, y+h), (x+w, y+2*h), (0, 0, 0), 2)
    # cv2.rectangle(img_c, (x_p, y_p), (x_pp, y_pp), (0, 0, 0), 2)
    # cv2.line(img_c, (x+w, y_pp), (x_pp, y+h), (0, 255, 0), 2)
    # cv2.line(img_c, (x, y_pp), (x_p, y+h), (0, 255, 0), 2)


    # cv2.imshow("12", img_c[y_p:y_pp, x_p:x_pp])
    # cv2.waitKey(0)

    mask1 = np.zeros_like(image)
    mask2 = np.zeros_like(image)

    def l1(p0, p1):
        return p0 * (y_pp - (y+h))/x - (p1 - (y+h))

    def l2(p0, p1):
        return (p0 - (x+w)) * (y+h - y_pp)/(x_pp - (x+w)) - (p1 - y_pp)

    xv, yv = np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0]))

    mask1 = np.ma.masked_less(l1(xv, yv), 0).mask
    mask2 = np.ma.masked_less(l2(xv, yv), 0).mask
    mask = mask1 | mask2
    image[mask] = 255

    return image[y_p:y_pp, x_p:x_pp]