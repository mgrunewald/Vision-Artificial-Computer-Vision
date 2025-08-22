import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import random

from parte4__CNN import SimpleCNN

script_dir = os.path.dirname(os.path.abspath(__file__))
split_dir = os.path.join(script_dir, 'splitting')
raster_dir = os.path.join(script_dir, 'rasterization')

X_test = np.load(os.path.join(split_dir, 'X_test.npy'), allow_pickle=True)
y_test = np.load(os.path.join(split_dir, 'y_test.npy'))
class_names = np.load(os.path.join(raster_dir, 'class_names.npy'), allow_pickle=True)

model = SimpleCNN(len(class_names))
model.load_state_dict(torch.load(os.path.join(script_dir, 'modelo_cnn.pth'), map_location=torch.device('cpu')))
model.eval()

X_train = np.load(os.path.join(split_dir, 'X_train.npy'), allow_pickle=True)
y_train = np.load(os.path.join(split_dir, 'y_train.npy'))
X_arr = np.stack([np.array(img, dtype=np.float32) for img in X_train])
y_arr = np.array(y_train)

heatmaps_por_clase = {}
for class_idx in range(len(class_names)):
    class_imgs = X_arr[y_arr == class_idx]
    mean_img = np.mean(class_imgs, axis=0)
    heatmaps_por_clase[class_idx] = mean_img

n = 10
random_indices = random.sample(range(len(X_test)), n)

for idx in random_indices:
    img_np = np.array(X_test[idx], dtype=np.uint8)
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output).item()

    pred_class_name = class_names[pred]
    heatmap = heatmaps_por_clase[pred]

    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap="hot", alpha=0.8)
    plt.imshow(img_np, cmap="gray", alpha=0.35)
    plt.title(f"Pred: {pred_class_name} (idx {idx})")
    plt.axis("off")
    plt.show()
