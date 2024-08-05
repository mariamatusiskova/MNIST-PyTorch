from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import SGD
# # Contains some utilities for working with the image data
# import torchvision
# neural network module of PyTorch
from torch.utils.data import Dataset, DataLoader, random_split

train_val_filepath = 'MNIST/processed/training.pt'
x_data, y_labels = torch.load(train_val_filepath)

# print(x_data)
# print(x.shape)
# print(x_data[2])
# print(x[2].shape)

# pixels in array of an image
# print(x[2].numpy())

# print(y_labels)
# print(y.shape)
# print(y[2])
# print(y[2].numpy())

plt.imshow(x_data[2].numpy())
plt.title(f'Number is {y_labels[2].numpy()}')
plt.colorbar()
plt.show()

## Loading the MNIST dataset
train_val_filepath = 'MNIST/processed/training.pt'
test_filepath = 'MNIST/processed/test.pt'


class MNISTDataset(Dataset):

    # fetching data from sample
    def __init__(self, filepath: str):
        self.x_data, self.y_labels = torch.load(filepath)
        # normalization of the input from [0-255] to [0-1]
        self.x_data = self.x_data.float()
        # num classes [0-9]
        # self.y_labels = F.one_hot(self.y_labels, num_classes=10).to(float)

    # return the size of dataset
    def __len__(self) -> int:
        # returns the number of samples
        return len(self.x_data)
        # return self.x_data.shape[0]

    def __getitem__(self, index: int) -> tuple[Any, Tensor]:
        # returns the data and label for a given index
        x = self.x_data[index].unsqueeze(0)
        y = self.y_labels[index]
        return x, y


train_val_dataset = MNISTDataset(train_val_filepath)
test_dataset = MNISTDataset(test_filepath)

# split dataset into train and val
train_dataset, val_dataset = random_split(train_val_dataset, [50000, 10000])

## PyTorch DataLoader Object
# DataLoader - iteration with provided batch size
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# just trying model
# model = nn.Linear(28**2, 10)
# print(model.weight.shape)
# print(model.weight)
# print(model.bias.shape)
# print(model.bias)
# for images, labels in train_iterations:
#     print(labels)
#     print(images.shape)
#     break

## Model
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input 1 channel --> grayscale, layer will learn 10 different filters, kernel size 5x5 matrix
        # stride = 1, to go pixel by pixel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout2d(p=0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=20*4*4, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x: Any) -> Tensor:
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        #print(f"Input shape: {x.shape}")
        x = F.relu(self.conv2(x))
        #print(f"Shape after conv1: {x.shape}")
        x = self.max_pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        # # print(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print(x)
        return x

model = MNISTModel()
# lr - learning rate
optimization = SGD(model.parameters(), lr=0.001)


# print(model.linear1.weight.shape, model.linear1.bias.shape)
# print(model.linear2.weight.shape, model.linear2.bias.shape)
# list(model.parameters())
# outputs = ""
# for images, labels in train_iterations:
#     outputs = model(images)
#     break

# print('Outputs shape: ', outputs.shape)
# print('Sample outputs: \n', outputs[:2].data)

## Softmax function
# apply softmax to each tow in the outputs
# probabilities = F.softmax(outputs, dim=1)
# print("Sample probabilities:\n", probabilities[:2].data)

# print("Sum: ", torch.sum(probabilities[0]).item())
# max_probabilities, predictions = torch.max(probabilities, dim=1)
# print("\n")
# print(predictions)
# print("\n")
# print(max_probabilities)

## Training
def train_model(loader: DataLoader, dt_model: MNISTModel) -> float:
    # set the model to training mode
    model.train()
    loss_fun = nn.CrossEntropyLoss()
    train_loss = 0
    for x_data, y_labels in loader:
       # print(f"Test: Batch input shape: {x_data.shape}")
        # generate predictions
        predictions = dt_model(x_data)
        # calculate loss
        loss = loss_fun(predictions, y_labels)
        # reset gradients
        optimization.zero_grad()
        # update weights and biases, compute gradients
        loss.backward()
        optimization.step()
        # save data
        # extract the actual numerical value from the tensor
        train_loss += loss.item()
    return train_loss


def val_model(loader: DataLoader, dt_model: MNISTModel) -> (float, float):
    dt_model.eval()
    loss_fun = nn.CrossEntropyLoss()
    val_loss = 0
    is_correct = 0
    # not need to compute gradients
    with torch.no_grad():
        for x_data, y_labels in loader:
           # print(f"Val: Batch input shape: {x_data.shape}")
            # generate predictions
            raw_predictions = dt_model(x_data)
            # calculate loss
            loss = loss_fun(raw_predictions, y_labels)
            # calculate accuracy
            # argmax - finds the index of the maximum value,
            #          assign the class the highest predicted
            #          score for each input in the batch
            predictions = raw_predictions.argmax(dim=1)
            # compare predictions with real labels, then sum up
            # True, False and converts tensor to Pyton scalar
            is_correct += predictions.eq(y_labels).sum().item()
            val_loss += loss.item()
        return is_correct, val_loss


def compute_eval_data(train_loss, val_loss, is_correct) -> (float, float, float):
    eval_train_loss = train_loss / len(train_loader.dataset)
    eval_val_loss = val_loss / len(val_loader.dataset)
    eval_val_acc = is_correct / len(val_loader.dataset)
    return eval_train_loss, eval_val_loss, eval_val_acc


def fit(dt_model: MNISTModel, epochs: int) -> (list, list, list):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        train_loss = train_model(train_loader, dt_model)
        is_correct, val_loss = val_model(val_loader, dt_model)
        eval_train_loss, eval_val_loss, eval_val_acc = compute_eval_data(train_loss, val_loss, is_correct)

        train_losses.append(eval_train_loss)
        val_losses.append(eval_val_loss)
        val_accuracies.append(eval_val_acc)

        print(f'Epoch {epoch + 1}/{epochs}',
              f'train loss: {eval_train_loss:.4f}',
              f'val loss: {eval_val_loss:.4f}',
              f'val acc: {eval_val_acc:.2%}',
              sep=' | '
              )

    return train_losses, val_losses, val_accuracies

epochs = 10
train_losses, val_losses, val_accuracies = fit(model, epochs)

# Plotting the results
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')
plt.show()


## Testing
def test_model(loader: DataLoader, dt_model: MNISTModel) -> (float, float):
    dt_model.eval()
    loss_fun = nn.CrossEntropyLoss()
    test_loss = 0
    is_correct = 0
    with torch.no_grad():
        for x_data, y_labels in loader:
           # print(f"Test: Batch input shape: {x_data.shape}")
            # generate predictions
            raw_predictions = dt_model(x_data)
            # calculate loss
            loss = loss_fun(raw_predictions, y_labels)
            # calculate accuracy
            predictions = raw_predictions.argmax(dim=1)
            is_correct += predictions.eq(y_labels).sum().item()
            test_loss += loss.item()
    return is_correct, test_loss

# Evaluate the model on the test dataset
is_correct, test_loss = test_model(test_loader, model)

# Compute evaluation metrics
eval_test_loss = test_loss / len(test_loader.dataset)
eval_test_acc = is_correct / len(test_loader.dataset)

print(f'Test Loss: {eval_test_loss:.4f} | Test Accuracy: {eval_test_acc:.2%}')



## Visualize the results

figure, axes = plt.subplots(10, 5, figsize=(10, 15))

for i, (x_d, y_d) in enumerate(test_loader):
    if i >= 5:
        break

    # make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(x_d).argmax(dim=1)

    # loop through the batch of images
    for j in range(x_d.size(0)):
        # create a subplot for each image
        ax = plt.subplot(10, 5, i * 10 + j + 1)
        # extract a single image from the batch and remove the channel dimension
        img = x_d[j].squeeze(0)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {predictions[j].item()}')

# Adjust the layout of the figure
figure.tight_layout()
plt.show()


# naprogramovat unet -> najst data Toy dataset segmentacia
# vision transformer

