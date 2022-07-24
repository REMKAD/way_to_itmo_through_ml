import numpy as np
import cv2
import matplotlib.pyplot as plt


# считываем решение и переводим его в двоичный вид
img = cv2.imread('img\po.jpg')
img_binary = cv2.threshold(img, 145, 255, cv2.THRESH_BINARY)[1]
# разбиваем цельное решение на отдельные строчки и записываем картинки отдельных строчек в список
t = False
gate = 0
data = []
plt.imshow(img_binary)
plt.show()
x = 0
y = 0
while y != img_binary.shape[0]:
    if img_binary[y, x].tolist() != [255, 255, 255]:
        t = True
        gate = y
        while t:
            for x_1 in range(img_binary.shape[1]):
                if img_binary[y, x_1].tolist() != [255, 255, 255]:
                    y += 1
                    break
            else:
                t = False
                if abs(gate-y) > 25:
                    data.append(img[gate:y, 0:img_binary.shape[1]])
    else:
        if x != img_binary.shape[1]-1:
            x += 1
        else:
            y += 1
            x = 0
        t = False

# вывод каждой отдельной строчки
for i in range(len(data)):
    cv2.imshow('', data[i])
    cv2.waitKey(0)


