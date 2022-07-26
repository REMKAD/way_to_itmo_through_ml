from main import get_val_transforms, torch, Tokenizer, predict, config_json, CRNN
import numpy as np
import os
import cv2
import json
from matplotlib import pyplot as plt
from sympy import Symbol, solve, sqrt, preorder_traversal, simplify, Eq, solveset, sin

# функции для предсказания
class InferenceTransform:
    def __init__(self, height, width):
        self.transforms = get_val_transforms(height, width)

    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor


class OcrPredictor:
    def __init__(self, model_path, config, device='cpu'):
        self.tokenizer = Tokenizer(config['alphabet'])
        self.device = torch.device(device)
        # load model
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config['image']['height'],
            width=config['image']['width'],
        )

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, "
                            f"tuple or list, found {type(images)}.")

        images = self.transforms(images)
        pred = predict(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred

# импортируем обученную модель
predictor = OcrPredictor(
    model_path='new_data/model-13-0.0538.ckpt',
    config=config_json
)

# считываем решение

img = cv2.imread('img\img.png')
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
                    data.append(img[gate:y, 0:img_binary.shape[1]].copy())
    else:
        if x != img_binary.shape[1]-1:
            x += 1
        else:
            y += 1
            x = 0
        t = False

# разбиваем посимвольно
data_el = []
k = 0
for i in range(len(data)):
    data_el.append([])
    x = 0
    y = 0
    gate = 0
    img_binary = cv2.threshold(data[i], 145, 255, cv2.THRESH_BINARY)[1]
    while x != data[i].shape[1]:
        if img_binary[y, x].tolist() != [255, 255, 255]:
            t = True
            gate = x
            while t:
                for y_1 in range(data[i].shape[0]):
                    if img_binary[y_1, x].tolist() != [255, 255, 255]:
                        x += 1
                        break
                else:
                    t = False
                    if abs(gate - x) > 5:
                        data_el[k].append(data[i][0:img_binary.shape[0], gate:x].copy())

        else:
            if y != img_binary.shape[0] - 1:
                y += 1
            else:
                x += 1
                y = 0
            t = False

    k += 1
# создаем словарь предсказаний
pred_json = {}
print_images = True
predskaz = []
# считываем строчки и делаем по ним предсказания
for j in range(len(data_el)):
    predskaz.append('')
    for i in range(len(data_el[j])):
        pred = predictor(data_el[j][i])
        pred_json[i] = pred

        if print_images:
            img = cv2.cvtColor(data_el[j][i], cv2.COLOR_BGR2RGB)

            print('Prediction: ', predictor(img))
            predskaz[j] += predictor(img)
            print(predskaz)

print(predskaz)
with open('prediction_HTR.json', 'w') as f:
    json.dump(pred_json, f)


x = Symbol('x')
sol = solveset(predskaz[0].simplify(), x)
for i in range(len(predskaz)):
    if solveset(predskaz[i].simplify(), x) != sol:
        print(f'eror in line{i+1}')



