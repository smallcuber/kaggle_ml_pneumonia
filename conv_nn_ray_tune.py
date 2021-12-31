#%%
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchsummary import summary
import os

from ray import tune, init
from ray.tune.schedulers import ASHAScheduler
import streamlit as st
import numpy as np
#%%
# noinspection PyTypeChecker
class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 3: colors - R G B, 6: output layer size, 5: convolution kernel size
        self.pool = nn.MaxPool2d(4, 4)  #4: Pool size, 4: Stride size
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

        self.model_feature_learning = torch.nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool
        )

        self.model_classification = torch.nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):  # n: number samples in a batch.
        # Start with n, 3, 32, 32
        x = self.model_feature_learning(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.model_classification(x)
        return x

# List of pre-trained models for fine-tuning
models = {
    'dense121' : torchvision.models.densenet121(pretrained=True, progress=True),
    'tf_effinet_b4': torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b4_ns', pretrained=True),
    'effinet_b3': torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b3', pretrained=True)
}

def data_loader_cnn(_data_dir):
    _test_dir = os.path.join(data_dir, 'test')
    _train_dir = os.path.join(data_dir, 'train')
    _val_dir = os.path.join(data_dir, 'val')

    _transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor()])
    _dataset_test = datasets.ImageFolder(_test_dir, transform=_transform)
    _dataset_train = datasets.ImageFolder(_train_dir, transform=_transform)
    _dataset_val = datasets.ImageFolder(_val_dir, transform=_transform)

    _loader_test = DataLoader(_dataset_test, batch_size=batch_size, shuffle=True)
    _loader_train = DataLoader(_dataset_train, batch_size=batch_size, shuffle=True)
    _loader_val = DataLoader(_dataset_val, batch_size=batch_size, shuffle=True)

    return _loader_test, _loader_train, _loader_val, _dataset_test, _dataset_train, _dataset_val

def data_loader_fine_tune(_data_dir):
    _test_dir = os.path.join(_data_dir, 'test')
    _train_dir = os.path.join(_data_dir, 'train')
    _val_dir = os.path.join(_data_dir, 'val')

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.RandomRotation(degrees=(-20, +20)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_test_val = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    _dataset_test = datasets.ImageFolder(_test_dir, transform=transform_test_val)
    _dataset_train = datasets.ImageFolder(_train_dir, transform=transform)
    _dataset_val = datasets.ImageFolder(_val_dir, transform=transform_test_val)

    _loader_test = DataLoader(_dataset_test, batch_size=batch_size, shuffle=True)
    _loader_train = DataLoader(_dataset_train, batch_size=batch_size, shuffle=True)
    _loader_val = DataLoader(_dataset_val, batch_size=batch_size, shuffle=True)

    return _loader_test, _loader_train, _loader_val, _dataset_test, _dataset_train, _dataset_val

# writer = SummaryWriter('runs/mnist')
# Use local GPU for CNN models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
# Set ML configuration

# Configure Streamlit layout
# c1, c2 = st.beta_columns((1, 2))
# b1 = st.image

# Pick image size
# image_size = c1.number_input('Choose the size of training image, between 0 - 2000', min_value=0, max_value=2000, step=1, value=500)

config = {
    "lr" : 0.01,
    "momentum" : 1,
}

image_size = 300
# c1.write(f'Selected {image_size} as image for the model')
# Image directory, linked to selected image size
data_dir =  os.path.abspath(rf'../Pneumonia_classification_data/reshape_{image_size}')
# Pick number of epochs
# num_epochs = c1.number_input('Choose the number of epochs for training.', min_value=0, step=1, value=10)
# c1.write(f'Selected {num_epochs} as the training epochs')
num_epochs = 50
# Pick batch size
# batch_size = c1.number_input('Choose the number of batch size for training.', min_value=0, step=1, value=4)
# c1.write(f'Selected {batch_size} as the training batch size')
batch_size = 4
# Pick learning rate
# rate_learning = c1.number_input('Choose the number of batch size for training.', min_value=0.0, value=0.001)
rate_learning = 0.01
# Available classes
classes = ('Normal', 'Pneumonia')
#%%
# Load and transform datasets
# Images are processed from main.py to 300x300 greyscale jpeg format
loader_test, loader_train, loader_val, dataset_test, dataset_train, dataset_val = data_loader_cnn(_data_dir=data_dir)

# Check loaded data samples
images, labels = next(iter(loader_train))
# print(images[0])
# print(images.shape)
# plt.imshow(torchvision.utils.make_grid(images).permute(1,2,0))
# print(f'The answer of images are {labels}')
# lst = [x.item() for x in labels]
# new_dict = dict((v, k) for k, v in dataset_test.class_to_idx.items())
# new_lst = [new_dict.get(item) for item in lst]
# print(new_lst)
#%%


def train(_model, _optimizer, _loader_train):
    _model.train()
    for batch_dix, (_images, _labels) in enumerate(_loader_train):

        _images, _labels = _images.to(device), _labels.to(device)
        _optimizer.zero_grad()
        _output = _model(_images)
        _loss = F.nll_loss(_output, _labels)
        _loss.backward()
        _optimizer.step()


def test(_model, _loader_test):
    _model.eval()
    _correct = 0
    _total = 0
    with torch.no_grad():
        for batch_dix, (_images, _labels) in enumerate(_loader_test):
            _images, _labels = _images.to(device), _labels.to(device)
            _output = _model(_images)
            _, _predicted = torch.max(_output.data, 1)
            _total += _labels.size(0)
            _correct += (_predicted == _labels).sum().item()
        return _correct / _total


def train_dense(_config=dict):
    _model = ConvolutionalNeuralNet().to(device)
    _optimizer = torch.optim.SGD(_model.parameters(), lr=_config['lr'], momentum=_config["momentum"])
    for _i in range(num_epochs):
        train(_model, _optimizer, loader_train)
        _acc = test(_model, loader_test)
        tune.report(mean_accuracy=_acc)

        torch.save(_model.state_dict(), "./model_state.pth")

#%%
# Setup model, loss, optimizer and total training steps
# model = ConvolutionalNeuralNet().to(device)
model = models['dense121'].to(device)
# print(model)
summary(model, input_size=(3, image_size, image_size))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=rate_learning)

# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1285*num_epochs)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1285*2, gamma=0.7)
n_total_steps = len(loader_train)
#%%
search_space = {
    "lr" : tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    "momentum" : tune.uniform(0.1, 0.9)
}

analysis = tune.run(train_dense, config=search_space, resources_per_trial={'gpu' : 1})
dfs = analysis.trial_dataframes
[d.mean_accuracy.plot() for d in dfs.values()]
#%%