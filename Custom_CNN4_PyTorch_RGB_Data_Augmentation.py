import math

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

import Utilities
from Preprocessing import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CustomCNN4(nn.Module):
    def __init__(self, in_channels=100, num_classes=5):
        super(CustomCNN4, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=2),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.4),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.4),

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            torch.nn.Flatten(),
            torch.nn.Linear(1600, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


# training function
def train_batch(model, criterion, optimizer, x, y):
    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    y_hat = model.forward(x)
    loss = criterion(y_hat, y)

    # Backward
    loss.backward()

    # Update parameters
    optimizer.step()

    return loss.data.item()


# make predictions
def predict(model, x):
    with torch.no_grad():
        outputs = model(x)
    predicted = torch.argmax(outputs.data, dim=1)
    return predicted.cpu().detach().numpy()


def run_cnn4(
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
        epochs=35,
        batch_size=8,
        learning_rate=0.001
):
    model = CustomCNN4(in_channels=100, num_classes=5)
    model = model.to(device=device)

    # for plotting
    plot_train_loss = []
    plot_val_loss = []
    plot_train_accuracy = []
    plot_val_accuracy = []
    y_val_predicted = None

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # run the training
    num_examples = X_train.shape[0]
    num_batches = math.ceil(num_examples / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            train_loss += train_batch(
                model,
                criterion,
                optimizer,
                X_train_tensor[start:end],
                y_train_tensor[start:end]
            )
        train_loss = train_loss / num_batches
        plot_train_loss.append(train_loss)

        model.eval()

        y_train_predicted = predict(model, X_train_tensor)
        y_train = y_train_tensor.cpu().detach().numpy()
        train_accuracy = accuracy_score(y_train, y_train_predicted) * 100
        plot_train_accuracy.append(train_accuracy)

        with torch.no_grad():
            y_val_hat = model(X_val_tensor)
        val_loss = criterion(y_val_hat, y_val_tensor).data.item()
        plot_val_loss.append(val_loss)

        y_val_predicted = predict(model, X_val_tensor)
        y_val = y_val_tensor.cpu().detach().numpy()
        val_accuracy = accuracy_score(y_val, y_val_predicted) * 100
        plot_val_accuracy.append(val_accuracy)

        print('Epoch %02d, train loss = %f, val loss = %f, train accuracy = %.2f%%, val accuracy = %.2f%%'
              % (epoch, train_loss, val_loss, train_accuracy, val_accuracy))

    print()

    plt.plot(plot_train_loss, label='Train Loss')
    plt.plot(plot_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(plot_train_accuracy, label='Train Accuracy')
    plt.plot(plot_val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return model, y_val_predicted


if __name__ == '__main__':
    grayscale = False
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
        base_path='./dataset',
        grayscale=grayscale,
        data_augmentation=True
    )

    X_train_tensor = torch.from_numpy(X_train).float().to(device=device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device=device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device=device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device=device)
    y_val_tensor = torch.from_numpy(y_val).long().to(device=device)

    model, y_val_predicted = run_cnn4(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
    print('Validation Set metrics')
    Utilities.display_metrics(y_val, y_val_predicted)

    y_test_predicted = predict(model, X_test_tensor)
    print('Test Set metrics')
    Utilities.display_metrics(y_test, y_test_predicted)
