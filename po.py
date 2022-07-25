from main import get_val_transforms, torch, Tokenizer, predict, config_json, CRNN
import numpy as np
import os
import cv2
import json
from matplotlib import pyplot as plt






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


predictor = OcrPredictor(
    model_path='new_data/model-49-0.0505.ckpt',
    config=config_json
)


pred_json = {}

count = 0

for img_name in os.listdir('img/'):
    img = cv2.imread(f'img/{img_name}')

    pred = predictor(img)
    pred_json[img_name] = pred

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()
    print('Prediction: ', predictor(img))



with open('prediction_HTR.json', 'w') as f:
    json.dump(pred_json, f)