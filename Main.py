import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def show(img):
    plt.imshow(img)
    plt.show()


image = cv2.imread('images/3.bmp')

text_image = np.zeros((150, 150, 3))
# text_image = np.zeros(image.shape)
text_image[50:90, :] = 100
text_image = cv2.putText(text_image, 'Label', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

text_image = np.tile(text_image, (int(np.ceil(image.shape[0] / 150)) * 2,
                                  int(np.ceil(image.shape[1] / 150)) * 2, 1))

rotation_matrix = cv2.getRotationMatrix2D(tuple(np.array(text_image.shape[1::-1]) / 2), random.uniform(-45, 45), 1.0)
text_image = cv2.warpAffine(text_image, rotation_matrix, text_image.shape[1::-1], flags=cv2.cv2.INTER_LINEAR)

text_image = text_image[int(text_image.shape[0] / 4):int(text_image.shape[0] / 4) + image.shape[0],
             int(text_image.shape[1] / 4):int(text_image.shape[1] / 4) + image.shape[1]]
image = image.astype('float64')
cv2.addWeighted(image, 0.7, text_image, 0.3, 0, image)

cv2.imwrite('images/result.png', image)
