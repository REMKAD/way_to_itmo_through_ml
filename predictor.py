from main import get_val_transforms, torch, Tokenizer, predict, config_json, CRNN
import numpy as np
import os
import cv2
import json
from matplotlib import pyplot as plt


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
    model_path='new_data/model-4-0.0664.ckpt',
    config=config_json
)

# считываем изображение решения и переводим его в двоичный вид
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
pred_json = {}

count = 0
print_images = True
for img_name in os.listdir('img/'):
    img = cv2.imread(f'img/{img_name}')

    pred = predictor(img)
    pred_json[img_name] = pred

    if print_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        print('Prediction: ', predictor(img))
        count += 1

    if count > 3:
        print_images = False

with open('prediction_HTR.json', 'w') as f:
    json.dump(pred_json, f)