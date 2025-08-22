import os
from matplotlib.colors import LinearSegmentedColormap
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchsummary import summary

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'modelo_cnn.pth')
split_dir  = os.path.join(script_dir, 'splitting')
cm_file    = os.path.join(split_dir, 'cm_normalized.npy')
fig_file   = os.path.join(script_dir, 'confusion_matrix_normalized.png')

class_names = np.load(
    os.path.join(script_dir, 'rasterization', 'class_names.npy'),
    allow_pickle=True
)
X_test = np.load(os.path.join(split_dir, 'X_test.npy'), allow_pickle=True)
y_test = np.load(os.path.join(split_dir, 'y_test.npy'))

X_test_arr    = np.stack([np.array(img, dtype=np.uint8) for img in X_test])
X_test_tensor = torch.tensor(X_test_arr).unsqueeze(1).float() / 255.0
y_test_tensor = torch.tensor(y_test)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3, padding=1),  nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),                nn.Dropout(dropout),

            nn.Conv2d(32,64,3, padding=1),  nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3, padding=1),  nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                nn.Dropout(dropout),

            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1,1,32,32)
            out   = self.body(dummy)
            feat_size = out.view(1,-1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(feat_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(class_names), dropout=0.3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

summary(model, input_size=(1,32,32))

y_true = []
y_pred = []

with torch.no_grad():
    for i in range(len(X_test_tensor)):
        img   = X_test_tensor[i].unsqueeze(0).to(device)
        label = y_test_tensor[i].item()
        out   = model(img)
        pred  = torch.argmax(out, dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

print(classification_report(y_true, y_pred, target_names=class_names))

if os.path.exists(cm_file):
    cm_norm = np.load(cm_file)
else:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
    np.save(cm_file, cm_norm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
yellow_cmap = LinearSegmentedColormap.from_list("custom_yellow", ["#ffffff", "#FFE008"])
disp.plot(
    ax=ax,
    xticks_rotation='vertical',
    cmap=yellow_cmap,
    values_format='.2f'
)

# Cambiar el color del texto a negro manualmente
for text in disp.text_.ravel():
    text.set_color("black")
ax.set_title("Matriz de Confusión Normalizada")
plt.tight_layout()
fig.savefig(fig_file, dpi=150)
plt.show()

indices = np.random.choice(len(X_test_tensor), size=10, replace=False)
for i in indices:
    img        = X_test_tensor[i].squeeze(0).cpu().numpy()
    true_label = y_test_tensor[i].item()
    with torch.no_grad():
        pred_label = torch.argmax(model(X_test_tensor[i].unsqueeze(0).to(device)), dim=1).item()

    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {class_names[pred_label]}\nReal: {class_names[true_label]}")
    plt.axis('off')
    plt.show()
