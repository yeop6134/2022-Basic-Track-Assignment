import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


HYPER_PARMATER = {
    'IMAGE_SIZE': 128,
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 1e-3,
    "BATCH_SIZE": 4,
    "SEED": 28
}


def set_seed(seed: int) -> None:
    random.seed()
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.randome.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(HYPER_PARMATER["SEED"])

df = pd.read_csv("./data/train.csv")

df['label'][df['label'] == '10-1'] = 10  # label : 10-1 -> 10
df['label'][df['label'] == '10-2'] = 0  # Label : 10-2 -> 0
df['label'] = df['label'].apply(lambda x: int(x))  # Dtype : object -> int


def get_data(data_dir, is_train=True):
    img_path_list = []
    label_list = []

    img_path_list.extend(glob(os.path.join(data_dir, "*.png")))
    img_path_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    label_list.extend[df['label']]

    if is_train:
        return img_path_list, label_list

    else:
        return img_path_list


all_img_path, all_label = get_data('./data/train', True)
test_img_path = get_data('./data/test', False)


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None):
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)


train_len = int(len(all_img_path)*0.75)
val_len = int(len(all_img_path)*0.25)

train_img_path = all_img_path[:train_len]
train_label = all_label[:train_len]

val_img_path = all_img_path[:val_len]
val_label = all_label[:val_len]

train_transform = transforms.Compose([
    transforms.ToPILImage(),  # Numpy배열에서 PIL이미지로
    transforms.Resize([HYPER_PARMATER['IMG_SIZE'],
                      HYPER_PARMATER['IMG_SIZE']]),  # 이미지 사이즈 변형
    transforms.ToTensor(),  # 이미지 데이터를 tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 이미지 정규화

])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([HYPER_PARMATER['IMG_SIZE'],
                      HYPER_PARMATER['IMG_SIZE']]),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Get Dataloader

# CustomDataset class를 통하여 train dataset생성
train_dataset = CustomDataset(
    train_img_path, train_label, train_mode=True, transforms=train_transform)
# 만든 train dataset를 DataLoader에 넣어 batch 만들기
train_loader = DataLoader(
    train_dataset, batch_size=HYPER_PARMATER['BATCH_SIZE'], shuffle=True, num_workers=0)  # BATCH_SIZE : 24

# vaildation 에서도 적용
vali_dataset = CustomDataset(
    val_img_path, val_label, train_mode=True, transforms=test_transform)
vali_loader = DataLoader(
    vali_dataset, batch_size=HYPER_PARMATER['BATCH_SIZE'], shuffle=False, num_workers=0)

train_batches = len(train_loader)
vali_batches = len(vali_loader)

print('total train imgs :', train_len, '/ total train batches :', train_batches)
print('total valid imgs :', val_len, '/ total valid batches :', vali_batches)


train_features, train_labels = next(
    iter(train_loader))

img = train_features[0]
label = train_labels[0]
plt.imshow(img[0], cmap="gray")
plt.show()
print(f"Label: {label}")


class CNNclassification(torch.nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.fc_layer = nn.Sequential(
            nn.Linear(3136, 11)  # fully connected layer(ouput layer)
        )

    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        # concacatenation
        x = torch.flatten(x, start_dim=1)

        out = self.fc_layer(x)
        return out


model = CNNclassification().to(device)
criterion = torch.nn.CrossEntropyLoss()

# Select Optimizer
optimizer = optim.SGD(
    params=model.parameters(), lr=HYPER_PARMATER["LEARNING_RATE"])
scheduler = None


def train(model, optimizer, train_loader, scheduler, device):
    model.to(device)
    n = len(train_loader)

    # Loss Function 정의
    criterion = nn.CrossEntropyLoss().to(device)
    best_acc = 0

    for epoch in range(1, HYPER_PARMATER["EPOCHS"]+1):
        # train
        model.train()
        running_loss = 0.0

        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device)  # 배치 데이터

            # initializtion
            optimizer.zero_grad()

            logit = model(img)  # 예측값 산출
            loss = criterion(logit, label)  # 손실함수 계산

            # Backpropagation
            loss.backward()

            # Optimization
            optimizer.step()
            running_loss += loss.item()

        print('[%d] Train loss: %.10f' %
              (epoch, running_loss / len(train_loader)))

        if scheduler is not None:
            scheduler.step()

        # Validation set 평가
        model.eval()
        vali_loss = 0.0
        correct = 0
        # validation 과정에서는 parameter update를 하지 않기 때문에 torch.no_grad() 사용
        with torch.no_grad():
            for img, label in tqdm(iter(vali_loader)):
                img, label = img.to(device), label.to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)

                pred = logit.argmax(dim=1, keepdim=True)

                correct += pred.eq(label.view_as(pred)).sum().item()
        vali_acc = 100 * correct / len(vali_loader.dataset)
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(
            vali_loader), correct, len(vali_loader.dataset), 100 * correct / len(vali_loader.dataset)))

        # Best model 저장
        if best_acc < vali_acc:
            best_acc = vali_acc

            torch.save(model.state_dict(), './saved/best_model.pth')
            print('Model Saved.')


train(model, optimizer, train_loader, scheduler, device)


def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred


test_dataset = CustomDataset(
    test_img_path, None, train_mode=False, transforms=test_transform)

test_loader = DataLoader(
    test_dataset, batch_size=HYPER_PARMATER['BATCH_SIZE'], shuffle=False, num_workers=0)

# Load Best accuracy model
checkpoint = torch.load('./saved/best_model.pth')
model = CNNclassification().to(device)
model.load_state_dict(checkpoint)

# Inference
preds = predict(model, test_loader, device)
preds[0:5]

# code for submission
submission = pd.read_csv('data/sample_submission.csv')
submission['label'] = preds

submission['label'][submission['label'] == 10] = '10-1'  # label : 10 -> '10-1'
submission['label'][submission['label'] == 0] = '10-2'  # Label : 0 -> '10-2'
submission['label'] = submission['label'].apply(
    lambda x: str(x))

submission.to_csv('submit.csv', index=False)
