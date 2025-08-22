import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from parte4__CNN import SimpleCNN

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'modelo_cnn.pth')
class_names = np.load(os.path.join(script_dir, 'rasterization', 'class_names.npy'), allow_pickle=True)
X_test = np.load(os.path.join(script_dir, 'splitting', 'X_test.npy'), allow_pickle=True)
y_test = np.load(os.path.join(script_dir, 'splitting', 'y_test.npy'))

model = SimpleCNN(len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

features = []
def forward_hook(module, input, output):
    features.append(output)
    output.retain_grad()  

target_layer = model.body[7]
target_layer.register_forward_hook(forward_hook)

n_imagenes = 10

for idx in tqdm(range(n_imagenes), desc="Generando Grad-CAM"):
    features.clear()

    img_np = np.array(X_test[idx])
    img_arr = np.array(img_np, dtype=np.uint8)
    img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    output = model(img_tensor)
    pred = torch.argmax(output)

    model.zero_grad()
    output[0, pred].backward()

    grads_val = features[0].grad[0].numpy()       
    activations = features[0].detach()[0].numpy()   
    weights = np.mean(grads_val, axis=(1, 2))      

    cam = np.sum(weights[:, None, None] * activations, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-6)

    #print(f"[{idx}] cam min: {cam.min():.4f}, max: {cam.max():.4f}, pred: {class_names[pred]}")

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(img_arr, cmap='gray')
    axs[0].set_title("Imagen original")
    axs[0].axis('off')

    axs[1].imshow(img_arr, cmap='gray')
    im = axs[1].imshow(cam, cmap='inferno', alpha=0.6)       
    axs[1].set_title(f"Grad-CAM - Pred: {class_names[pred]}")
    axs[1].axis('off')

    # agregamos la colorbar para entender qué es que está resaltando el Grad-CAM
    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Intensidad Grad-CAM")

    plt.tight_layout()
    plt.show()


