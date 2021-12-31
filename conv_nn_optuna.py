#%%
from time import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import os
import optuna
from optuna.trial import TrialState
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
    'dense121': torchvision.models.densenet121(pretrained=True, progress=True),
    'tf_effinet_b4': torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b4_ns', pretrained=True),
    'effinet_b3': torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b3', pretrained=True)
}


def data_loader_cnn(_data_dir):
    _test_dir = os.path.join(_data_dir, 'test')
    _train_dir = os.path.join(_data_dir, 'train')
    _val_dir = os.path.join(_data_dir, 'val')

    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
         transforms.RandomRotation(degrees=(-20, +20)),
         transforms.ToTensor()])

    transform_test_val = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor()])

    _dataset_test = datasets.ImageFolder(_test_dir, transform=transform_test_val)
    _dataset_train = datasets.ImageFolder(_train_dir, transform=transform)
    _dataset_val = datasets.ImageFolder(_val_dir, transform=transform_test_val)

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

config = {
    "lr": 0.01,
    "momentum": 1,
}

image_size = 300
# c1.write(f'Selected {image_size} as image for the model')
# Image directory, linked to selected image size
data_dir = os.path.abspath(rf'../Pneumonia_classification_data/reshape_{image_size}')
# Pick number of epochs
# num_epochs = c1.number_input('Choose the number of epochs for training.', min_value=0, step=1, value=10)
# c1.write(f'Selected {num_epochs} as the training epochs')
num_epochs = 20
# Pick batch size
# batch_size = c1.number_input('Choose the number of batch size for training.', min_value=0, step=1, value=4)
# c1.write(f'Selected {batch_size} as the training batch size')
batch_size = 4
# Pick learning rate
# rate_learning = c1.number_input('Choose the number of batch size for training.', min_value=0.0, value=0.001)
rate_learning = 0.01
# Available classes
classes = ('Normal', 'Pneumonia')
model_dir = os.path.abspath(r'../Pneumonia_classification')
#%%
# Load and transform datasets
# Images are processed from main.py to 300x300 greyscale jpeg format
# loader_test, loader_train, loader_val, dataset_test, dataset_train, dataset_val = data_loader_fine_tune(_data_dir=data_dir)

# Check loaded data samples
# images, labels = next(iter(loader_train))
# print(images[0])
# print(images.shape)
# plt.imshow(torchvision.utils.make_grid(images).permute(1,2,0))
# print(f'The answer of images are {labels}')
# lst = [x.item() for x in labels]
# new_dict = dict((v, k) for k, v in dataset_test.class_to_idx.items())
# new_lst = [new_dict.get(item) for item in lst]
# print(new_lst)
#%%


# def model_train(_model, _optimizer, _criterion, _loader_train):
#     _model.train()
#     for batch_dix, (_images, _labels) in enumerate(_loader_train):
#         _images, _labels = _images.to(device), _labels.to(device)
#         _optimizer.zero_grad()
#         _output = _model(_images)
#         _loss = _criterion(_output, _labels)
#         _loss.backward()
#         _optimizer.step()

def model_test(_model, _loader_test):
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


#%%
def train_trial(_model_selection, _scheduler_selection, _lr, _trial=None, _save_model=False):
    global data_dir
    # Setup model, loss, optimizer and total training steps
    writer = SummaryWriter('runs/mnist')
    if _model_selection == "CNN_custom":
        loader_test, loader_train, loader_val, dataset_test, dataset_train, dataset_val = \
            data_loader_cnn(_data_dir=data_dir)
    else:
        loader_test, loader_train, loader_val, dataset_test, dataset_train, dataset_val = \
            data_loader_fine_tune(_data_dir=data_dir)

    if _model_selection == "CNN_custom":
        model = ConvolutionalNeuralNet().to(device)
    elif _model_selection == "DENSENET121":
        model = models['dense121'].to(device)
    elif _model_selection == "EFFINET_B4":
        model = models['tf_effinet_b4'].to(device)
    elif _model_selection == "EFFINET_B3":
        model = models['effinet_b3'].to(device)

    # summary(model, input_size=(1, image_size, image_size))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=_lr)

    if _scheduler_selection == "COSINEANNELINGLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif _scheduler_selection == "STEPLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1285*num_epochs)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1285*2, gamma=0.7)
    n_total_steps = len(loader_train)

    for epoch in range(num_epochs):
        model.train()
        loss_per_epoch = 0.0
        for i, (_images, _labels) in enumerate(loader_train):
            _images, _labels = _images.to(device), _labels.to(device)
            optimizer.zero_grad()
            _output = model(_images)
            loss = criterion(_output, _labels)
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.item()
            if _scheduler_selection != "NONE":
                scheduler.step(epoch + (i / n_total_steps))
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.6e}')
        loss_per_epoch = loss_per_epoch / n_total_steps
        _accuracy = model_test(model, loader_test)
        print(f'Epoch {epoch + 1} Accuracy: {_accuracy: .6f}, Current Loss: {loss_per_epoch: .7e}')
        writer.add_scalar('Loss/train', loss_per_epoch, epoch)
        writer.add_scalar('Accuracy/train', _accuracy, epoch)
        if _trial:
            _trial.report(_accuracy, epoch)
            if _trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    print('Finished training')
    if _save_model:
        model_name = f'{_model_selection}_{_scheduler_selection}_{_lr}_{int(time())}.pth'
        torch.save(model.state_dict(), model_name)
        print(f'Best model saved as {model_name} with {num_epochs} epochs.')

    return _accuracy


def objective(trial):
    model_selection = trial.suggest_categorical("classifier",
                                                ["CNN_custom", "DENSENET121", "EFFINET_B4", "EFFINET_B3"])
    scheduler_selection = trial.suggest_categorical("scheduler",
                                                ["NONE", "COSINEANNELINGLR", "STEPLR"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    _accuracy = train_trial(trial, model_selection, scheduler_selection, lr, _save_model=False)
    return _accuracy


#%%


if __name__ == "__main__":
    study_trail = False
    study_name = "pneumonia"
    study = optuna.create_study(study_name=study_name,
                                storage='sqlite:///train_info.db',
                                load_if_exists=True,
                                direction="maximize")

    if study_trail:
        study.optimize(objective, n_trials=100)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Facts:")
        print("  Finished Trials: ", len(study.trials))
        print("  Pruned Trials: ", len(pruned_trials))
        print("  Complete Trials: ", len(complete_trials))

        print("Best Trial:")
        trial_best = study.best_trial
        print("  Value: ", trial_best.value)
        print("  Parameters: ")
        for key, value in trial_best.params.items():
            print("  {}: {}".format(key, value))
    else:
        # model = models['effinet_b3'].to(device)
        # model.load_state_dict(torch.load('EFFINET_B3_NONE_0.0894419081453405_1633060699.pth'))
        # loader_test, loader_train, loader_val, dataset_test, dataset_train, dataset_val = \
        #     data_loader_fine_tune(_data_dir=data_dir)
        # acc = model_test(_model=model, _loader_test=loader_test)
        # print(f'Best model with acc: {acc}')

        model_select = study.best_trial.params['classifier']
        scheduler_select = study.best_trial.params['scheduler']
        lr_select = study.best_trial.params['lr']
        num_epochs = 50  # Redefine epoch here
        accuracy = train_trial(_model_selection=model_select, _scheduler_selection=scheduler_select, _lr=lr_select,
                               _save_model=True)
