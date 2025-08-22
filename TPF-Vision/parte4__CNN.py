# primer entrenamiento con la CNN pero sin data augmentation 
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
split_dir = os.path.join(script_dir, 'splitting')

X_train = np.load(os.path.join(split_dir, 'X_train.npy'), allow_pickle=True)
y_train = np.load(os.path.join(split_dir, 'y_train.npy'))
X_val = np.load(os.path.join(split_dir, 'X_val.npy'), allow_pickle=True)
y_val = np.load(os.path.join(split_dir, 'y_val.npy'))

X_train_arr = np.stack([np.array(img, dtype=np.uint8) for img in X_train])
X_val_arr = np.stack([np.array(img, dtype=np.uint8) for img in X_val])
X_train_tensor = torch.tensor(X_train_arr).unsqueeze(1).float() / 255.0
X_val_tensor = torch.tensor(X_val_arr).unsqueeze(1).float() / 255.0

y_train_tensor = torch.tensor(y_train).long()
y_val_tensor = torch.tensor(y_val).long()

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
val_ds = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.25):
        super(SimpleCNN, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, X_train_tensor.shape[2], X_train_tensor.shape[3])
            out = self.body(dummy)
            self.flattened_size = out.view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
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

    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):

        model.train()
        total_train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        val_losses.append(total_val_loss / len(val_loader))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
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

    save_path = os.path.join(script_dir, 'modelo_cnn.pth')
    if best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f"Modelo guardado en: {save_path}")