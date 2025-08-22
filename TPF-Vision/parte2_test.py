#%%

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

images_path = os.path.join(script_dir, 'rasterization', 'images_list.npy')
labels_path = os.path.join(script_dir, 'rasterization', 'labels_list.npy')
classes_path = os.path.join(script_dir, 'rasterization', 'class_names.npy')

images = np.load(images_path, allow_pickle=True)
labels = np.load(labels_path, allow_pickle=True)
class_names = np.load(classes_path, allow_pickle=True)
#%%

i = 800000
# elegir i entre 0 y 1.565.838
print("Array de imagen:")
print(images[i])

print(f"\nLabel: {labels[i]}")
print(f"Clase: {class_names[labels[i]]}")


# %%
