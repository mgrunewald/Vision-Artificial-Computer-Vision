import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
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
            dummy = torch.zeros(1, 1, 32, 32) 
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

clases = ['apple', 'butterfly', 'cloud', 'eye', 'flower', 'ice cream', 'moon', 'rainbow', 'smiley face', 'star', 'sun', 'tree']
script_dir = os.path.dirname(os.path.abspath(__file__))
carpeta = os.path.join(script_dir, "dibujos_amigos", "dibujos_dan")
modelo_path = os.path.join(script_dir, "modelo_cnn.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelo = SimpleCNN(num_classes=len(clases)).to(device)
modelo.load_state_dict(torch.load(modelo_path, map_location=device))
modelo.eval()

def procesar_imagen_como_raster(input_path, final_size=32, umbral=80):
    img = Image.open(input_path).convert("L")
    img_np = np.array(img)
    bin_img = (img_np < umbral).astype(np.uint8) * 255
    coords = np.argwhere(bin_img > 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = bin_img[y0:y1, x0:x1]

    h, w = cropped.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = cropped

    img_final = Image.fromarray(padded).resize((final_size, final_size), Image.Resampling.LANCZOS)
    return img_final

archivos = [f for f in os.listdir(carpeta) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
imagenes = []
titulos = []

for nombre_archivo in tqdm(archivos, desc="Clasificando imágenes"):
    path = os.path.join(carpeta, nombre_archivo)
    img_proc = procesar_imagen_como_raster(path)
    img_tensor = torch.tensor(np.array(img_proc)).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        salida = modelo(img_tensor)
        pred_idx = salida.argmax(dim=1).item()
        pred_clase = clases[pred_idx]
        imagenes.append(img_proc)
        titulos.append(f"{pred_clase}")

cols = 4
rows = (len(imagenes) + cols - 1) // cols
plt.figure(figsize=(3 * cols, 3 * rows))
for i, (img, titulo) in enumerate(zip(imagenes, titulos)):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(titulo, fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.show()
