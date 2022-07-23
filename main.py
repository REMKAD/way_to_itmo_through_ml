# импортируем нужные библиотеки
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2
import os
import json
from matplotlib import pyplot as plt
import random as rnd
from torch import cuda
# создаем объект типа json в которой хранится data


d = {}
for i in os.listdir('data1set'):
    for j in os.listdir(f'data1set/{i}/'):
        d[j] = i.split('_')[0]
with open('ans.json', 'w') as f:
    f.write(json.dumps(d))

# перемешиваем data
with open('ans.json') as f:
    a = json.load(f)
b = {}
k = [i for i in a.keys()]
rnd.shuffle(k)
for i in k:
    b[i] = a[i]

# заменяем исходные данные на перемешанные
with open('ans.json', 'w') as f:
    f.write(json.dumps(b))

    # считываем data
with open('ans.json') as f:
    train_data = json.load(f)

    # выводим длину train len
train_data = [(k, v) for k, v in train_data.items()]
print('train len', len(train_data))

# разбиваем data на обучающую и валидационную
split_coef = 0.75
train_len = int(len(train_data) * split_coef)

train_data_splitted = train_data[:train_len]
val_data_splitted = train_data[train_len:]

print('train len after split', len(train_data_splitted))
print('val after split', len(val_data_splitted))

with open('train_ans_splitted.json', 'w') as f:
    json.dump(dict(train_data_splitted), f)

with open('val_ans_splitted.json', 'w') as f:
    json.dump(dict(val_data_splitted), f)

# 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_json = {
    "alphabet": "cosinx=-+1234567890()pimqrt",
    "save_dir": "new_data",
    "num_epochs": 5,
    "image": {
        "width": 256,
        "height": 32
    },
    "train": {
        "root_path": "data1set",
        "json_path": "train_ans_splitted.json",
        "batch_size": 64
    },
    "val": {
        "root_path": "data1set",
        "json_path": "val_ans_splitted.json",
        "batch_size": 128
    }
}



# реорганизация данных в batch
def collate_fn(batch):
    images, texts, enc_texts = zip(*batch)
    images = torch.stack(images, 0)
    text_lens = torch.LongTensor([len(text) for text in texts])
    enc_pad_texts = pad_sequence(enc_texts, batch_first=True, padding_value=0)
    return images, texts, enc_pad_texts, text_lens


def get_data_loader(transforms, json_path, root_path, tokenizer, batch_size, drop_last):
    dataset = OCRDataset(json_path, root_path, tokenizer, transforms)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=1
    )
    return data_loader


class OCRDataset(Dataset):
    def __init__(self, json_path, root_path, tokenizer, transform=None):
        super().__init__()
        self.transform = transform
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.data_len = len(data)

        self.img_paths = []
        self.texts = []
        for img_name, text in data.items():
            self.img_paths.append(os.path.join(root_path, img_name))
            self.texts.append(text)
        self.enc_texts = tokenizer.encode(self.texts)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        ############################
        ########################
        #######################
        #######################
        ##########################
        ###########################
        img_path = self.img_paths[idx].split('\\')[1]

        with open('ans.json') as ff:
            meaw = json.load(ff)
        img_path = self.img_paths[idx].split('\\')[0] + '\\' + meaw[img_path] + '\\' + img_path
        ##########################
        #########################
        ###########################
        ############################
        #########################
        ########################
        ##################

        text = self.texts[idx]
        enc_text = torch.LongTensor(self.enc_texts[idx])

        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, text, enc_text

    # класс для подсчёта элементов


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # токенайзер(вспомогательный класс, который преобразует текст в числа)


OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'


def get_char_map(alphabet):
    char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
    char_map[CTC_BLANK] = 0
    char_map[OOV_TOKEN] = 1
    return char_map


class Tokenizer:

    def __init__(self, alphabet):
        self.char_map = get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    # из строк в числа

    def encode(self, word_list):
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[OOV_TOKEN]
                 for char in word]
            )
        return enc_words

    def get_num_chars(self):
        return len(self.char_map)

    # из числа в строку
    def decode(self, enc_word_list):
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                if (
                        char_enc != self.char_map[OOV_TOKEN]
                        and char_enc != self.char_map[CTC_BLANK]
                        and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
                dec_words.append(word_chars)
            return dec_words


# измеряет долю правильно предсказанных строк
def get_accuracy(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)
    avg_score = np.mean(scores)
    return avg_score


# аугментации для модели
class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class ImageResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        try:
            image = cv2.resize(image, (self.width, self.height))
            return image
        except cv2.error as e:
            print('Invalid frame!')



def get_train_transforms(height, width):
    transforms = torchvision.transforms.Compose([
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms


def get_val_transforms(height, width):
    transforms = torchvision.transforms.Compose([
        ImageResize(height, width),
        MoveChannels(to_channels_first=True),
        Normalize(),
        ToTensor()
    ])
    return transforms


# определение модели

def get_resnet34_backbone(pretained=True):
    ############################
    ########################
    #######################
    #######################
    ##########################
    ###########################
    ##########################
    #########################
    ###########################
    ############################
    #########################
    ########################
    ##################
    m = torchvision.models.resnet34(pretrained=True)
    ############################
    ########################
    #######################
    #######################
    ##########################
    ###########################
    ##########################
    #########################
    ###########################
    ############################
    #########################
    ########################
    ##################
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiSIM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(self, number_class_symbols, time_feature_count=256, lstm_hidden=256, lstm_len=2):
        super().__init__()
        self.feature_extractor = get_resnet34_backbone(pretained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((time_feature_count, time_feature_count))
        self.bilstm = BiSIM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        ############################
        ########################
        #######################
        #######################
        ##########################
        ###########################
        ##########################
        #########################
        ###########################
        ############################
        #########################
        ########################
        ##################
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        ############################
        ########################
        #######################
        #######################
        ##########################
        ###########################
        ##########################
        #########################
        ###########################
        ############################
        #########################
        ########################
        ##################
        return x


# обучение

def val_loop(data_loader, model, tokenizer, device):
    acc_avg = AverageMeter()
    for images, texts, _, _ in data_loader:
        batch_size = len(texts)
        text_preds = predict(images, model, tokenizer, device)
        acc_avg.update(get_accuracy(texts, text_preds), batch_size)

    ###########################3
    ############################
    #########################
    ##########################
    # Я НЕ ПОНИМАЮ, ЧТО ЭТО ЗА ПРИНТ, ПОЭТОМУ ЗАКОМЕНТИЛ, ЕСЛИ УБРАТЬ КОМЕНТ, ТО ПОЧЕМУ-ТО НЕ СРАБОТАЕТ
    #print(f'Validation, acc: {acc_avg:4f}')
    ###########################
    ########################
    ####################
    return acc_avg.avg



def train_loop(data_loader, model, criterion, optimizer, epoch):
    loss_avg = AverageMeter()
    model.train()
    for images, texts, enc_pad_texts, text_lens in data_loader:
        model.zero_grad()
        images = images.to(DEVICE)
        batch_size = len(texts)
        output = model(images)
        output_lenghts = torch.full(
            size=(output.size(1),),
            fill_value=output.size(0),
            dtype=torch.long
        )
        loss = criterion(output, enc_pad_texts, output_lenghts, text_lens)
        loss_avg.update(loss.item(), batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    print(f'\nEpoch {epoch}, Loss: {loss_avg.avg:.5f}, LR: {lr:.7f}')
    return loss_avg.avg


def predict(images, model, tokenizer, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
        #####################3
        #################
        ################
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    #####################
    ##################3
    ###############
    #################
    text_preds = tokenizer.decode(pred)
    return text_preds


def get_loaders(tokenizer, config):
    train_transforms = get_train_transforms(
        height=config['image']['height'],
        width=config['image']['width']
    )
    train_loader = get_data_loader(
        json_path=config['train']['json_path'],
        root_path=config['train']['root_path'],
        transforms=train_transforms,
        tokenizer=tokenizer,
        batch_size=config['train']['batch_size'],
        drop_last=True
    )
    val_transforms = get_val_transforms(
        height=config['image']['height'],
        width=config['image']['width']
    )
    val_loader = get_data_loader(
        json_path=config['val']['json_path'],
        root_path=config['val']['root_path'],
        transforms=train_transforms,
        tokenizer=tokenizer,
        batch_size=config['val']['batch_size'],
        drop_last=False
    )
    return train_loader, val_loader


def train(config):
    tokenizer = Tokenizer(config['alphabet'])
    os.makedirs(config['save_dir'], exist_ok=True)
    train_loader, val_loader = get_loaders(tokenizer, config)

    model = CRNN(number_class_symbols=tokenizer.get_num_chars())
    model.to(DEVICE)

    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=15)
    best_acc = np.inf

    acc_avg = val_loop(val_loader, model, tokenizer, DEVICE)
    scheduler.step(acc_avg)
    print()
    print(acc_avg, best_acc)
    for epoch in range(config['num_epochs']):
        loss_avg = train_loop(train_loader, model, criterion, optimizer, epoch)
        acc_avg = val_loop(val_loader, model, tokenizer, DEVICE)
        scheduler.step(acc_avg)
        if acc_avg > best_acc:
            best_acc = acc_avg
            model_save_path = os.path.join(
                config["save_dir"], f'model-{epoch}-{acc_avg:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            print('Model weights saved')
        best_acc = acc_avg
        model_save_path = os.path.join(config["save_dir"], f'model-{epoch}-{acc_avg:.4f}.ckpt')
        torch.save(model.state_dict(), model_save_path)
        print('Model weights saved')

if __name__=='__main__':
    train(config_json)


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
    def __init__(self, model_path, config, device='cuda'):
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
    model_path='new_data/model-4-0.0000.ckpt',
    config=config_json
)
count = 0
img = cv2.imread('img\ko.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
print('Prediction: ', predictor(img))
count += 1