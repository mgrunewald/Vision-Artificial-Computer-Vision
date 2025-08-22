# Le metemos data augmentation 
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
split_dir = os.path.join(script_dir, 'splitting')

X_train = np.load(os.path.join(split_dir, 'X_train.npy'), allow_pickle=True)
y_train = np.load(os.path.join(split_dir, 'y_train.npy'))
X_val = np.load(os.path.join(split_dir, 'X_val.npy'), allow_pickle=True)
y_val = np.load(os.path.join(split_dir, 'y_val.npy'))

X_train_arr = np.stack([np.array(img, dtype=np.uint8) for img in X_train])
X_val_arr = np.stack([np.array(img, dtype=np.uint8) for img in X_val])

y_train_tensor = torch.tensor(y_train).long()
y_val_tensor = torch.tensor(y_val).long()

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

train_ds = CustomImageDataset(X_train_arr, y_train_tensor, transform=train_transform)
val_ds = CustomImageDataset(X_val_arr, y_val_tensor, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.25):
        super(SimpleCNN, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout(dropout),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(dropout),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, X_train_arr.shape[1], X_train_arr.shape[2])
            out = self.body(dummy)
            self.flattened_size = out.view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.flattened_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

if __name__ == "__main__":

    n_classes = len(np.unique(y_train))
    model = SimpleCNN(num_classes=n_classes, dropout=0.3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):

        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                break

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(script_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()

    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(script_dir, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    plt.close()

    save_path = os.path.join(script_dir, 'modelo_cnn_final.pth')
    if best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f"Modelo guardado en: {save_path}")
